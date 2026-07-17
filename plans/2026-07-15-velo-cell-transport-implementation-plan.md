<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Velo Cell Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the loopback-only `TcpCellClient`/`TcpControllerTransport` cell↔controller wire with an official `ai-dynamo/velo` v0.5.0 transport behind the unchanged `CellClient`/`ControllerTransport` seam, and rework cell launch/registration so cells can be Kubernetes pods reached by one operator-hardcoded DNS:port (zero discovery), all behind a `velo` cargo feature.

**Architecture:** Keep every cellular *measurement* seam (`IssuanceAuthority`, `RecordsShard`, `MetricsHeartbeat`, static `owned_positions` partition, global-ordinal / concatenation merge) byte-for-byte unchanged. Swap only the wire (velo typed-unary/am_send handlers + rendezvous for the records-shard partition), the launch mechanism (a `CellLauncher` seam: local-subprocess-over-velo vs k8s-pod-no-spawn), and the registration/barrier (a `register` RPC that returns the `CellLaunchSpec` and counts cells to `cell_count`). Reaching the controller with unmodified velo uses one `resolve_controller_peer()` seam whose implementation (placeholder-address vs bootstrap-PeerInfo-fetch) is chosen by a first-task spike.

**Tech Stack:** Rust 2024, `velo = { git = "https://github.com/ai-dynamo/velo.git", tag = "v0.5.0", default-features = false }`, tokio `current_thread`, `serde`/`serde_json` (velo typed payloads are JSON), `rmp-serde` (partition body), hyper (already a dep; only if the bootstrap-fetch spike branch is chosen).

**Reference spec:** `specs/cellular.md`.
**Velo source for reading during implementation:** a clean v0.5.0 checkout is at `/tmp/velo-main` (crate layout `velo` + `velo-ext`; key files cited inline below).

## Global Constraints

- **Official velo, `ai-dynamo/main` lineage only.** `velo = { git = "https://github.com/ai-dynamo/velo.git", tag = "v0.5.0", default-features = false, optional = true }`. A **small upstream patch is acceptable ONLY as a PR against `ai-dynamo/main`** (never the user's legacy fork). If the spike (Task 1) chooses a mechanism needing such a patch, the PR is authored against `ai-dynamo/main` and the Cargo dep is temporarily pinned to that PR's `rev` (or a personal branch of `ai-dynamo/main` HEAD) until it merges, then repinned to a released tag. Do not pin to `ajcasagrande/velo` or any legacy branch.
- **`velo` cargo feature gates all velo code.** Without it: `cells=1` byte-for-byte unchanged; `cells>1` fails closed with a clear diagnostic. With it: full controller/cell/velo path.
- **Zero discovery.** No etcd / NATS / velo-discovery backend. Cells reach the controller from one operator-hardcoded DNS:port only (env `AIPERF_CELL_CONTROLLER_ADDR`, name unchanged).
- **Static partition retained.** Do not touch `owned_positions`, `CellularAutonomousIssuer`, `RecordsShardPartition`, the merges, or the byte-parity contract.
- **SPDX header** (`// SPDX-FileCopyrightText …` + `// SPDX-License-Identifier: Apache-2.0`) atop every new source file. `//!` module docs; `///` on every public item.
- **Thread-per-core / `!Send`** on the runner path; `current_thread` runtime + `LocalSet`. The cellular control plane is OFF the hot path — a handful of messages — so a small multi-thread runtime for the controller's velo I/O is acceptable (as `run_cellular` already uses today).
- **No `git stash`** (repo rule). Commit each task. Commit whole files (`git add <file>`), never `git add -p`. Commit on the current branch `worktree-velo-cell-transport`. Use `git commit --no-verify` and run guards manually (see below) to avoid the pre-commit auto-stash hazard.
- **Docs guard:** `python tools/check_docs_current.py` must pass before any commit that touches `specs/`. `python tools/check_agent_files_sync.py` must pass after the agent-file edits (Task 12).
- **Windows:** UDS is `#[cfg(unix)]`; Windows uses TCP-loopback. The `LocalLauncher` transport must compile on both.
- **Build/test commands:** `cargo build -p aiperf-cli --features velo`; `cargo test -p aiperf --features velo --lib`; `cargo test -p aiperf-cli --features velo`; `cargo clippy --all-targets --features velo`. A no-feature build (`cargo build -p aiperf-cli`) must still succeed.

---

## File Structure

**Create:**
- `rust/aiperf/src/cellular/transport/mod.rs` — the transport-neutral seam: `CellClient`, `ControllerTransport`, `CellMessage`, `CellTransportError`, plus the new control DTOs (`CellRegister`, `CellPartitionShip`, `CellAck`). (Moved out of today's single-file `transport.rs`.)
- `rust/aiperf/src/cellular/transport/velo_transport.rs` — `#[cfg(feature = "velo")]` velo impls: `VeloControllerTransport` (binds, registers the 3 handlers, exposes a merged `recv()`), `VeloCellClient` (register → spec, heartbeat, ship-partition), handler names, DTO wiring, rendezvous partition transfer.
- `rust/aiperf/src/cellular/transport/connect.rs` — `#[cfg(feature = "velo")]` the `resolve_controller_peer()` seam + `build_cell_velo()` / `build_controller_velo()` transport construction (UDS `#[cfg(unix)]` / TCP-loopback / TCP-bind). Implementation chosen by Task 1 spike.
- `rust/runner/src/cell_launcher.rs` — `CellLauncher` trait, `CellLaunchContext`, `CellHandle`, `LocalLauncher`, `K8sLauncher`, launcher selection.
- `rust/aiperf/examples/velo_cell_spike.rs` — Task 1 spike (deleted or kept as a doc example after).

**Modify:**
- `rust/aiperf/Cargo.toml` — optional `velo` dep + `velo` feature.
- `rust/runner/Cargo.toml` — `velo` feature forwarding to `aiperf/velo`; UDS/loopback deps if any.
- `rust/aiperf/src/cellular/mod.rs` — `pub mod transport` unchanged path; adjust re-exports (remove `Tcp*`, add `Velo*` under `#[cfg(feature="velo")]`).
- `rust/runner/src/cellular_controller.rs` — `run_cellular` uses the `CellLauncher` + `VeloControllerTransport` + barrier; the slice/merge/report body unchanged.
- `rust/runner/src/cellular_cell.rs` — `CellRecordsShipper` sends over velo; `CellLaunchSpec` fetched via `register`.
- `rust/runner/src/main.rs` — cell bootstrap from env + velo register (not stdin); controller path; fail-closed without `velo`.
- `rust/runner/src/lib.rs` — `mod cell_launcher;` and re-exports.
- `rust/e2e/tests/test_cellular.rs`, `rust/e2e/tests/test_graph_cellular.rs` — run over the velo `LocalLauncher`.
- The four agent files + canonical in-place updates to `specs/cellular.md`,
  `specs/README.md`, and `llms.txt` (Task 12).

**Delete:** `TcpCellClient`, `TcpControllerTransport`, `encode_frame`, `read_frame`, `MAX_FRAME_LEN`, and their tests from the old `transport.rs` (folded into the mod split).

---

## Task 1: Spike — prove discovery-free cell↔controller reachability against velo v0.5.0

**Purpose:** Decide the `resolve_controller_peer()` implementation before building on it. Three candidate mechanisms; the spike picks the cleanest that works end-to-end and the later tasks consume its choice through one function.

**Candidate mechanisms (evaluate in this order):**
- **A — placeholder-address, zero patch:** construct the controller `PeerInfo` from `host:port` + a placeholder `InstanceId` using *existing public* velo API. Cleanest if a public constructor exists.
- **A′ — placeholder-address, tiny upstream patch:** same as A but enabled by a **small PR against `ai-dynamo/main`** exposing a public peer-from-`host:port` constructor (e.g. re-export `WorkerAddressBuilder` or add `WorkerAddress::tcp(addr)` / `PeerInfo::from_tcp_addr`) — *or* an injectable `VeloBuilder::instance_id`. Allowed per the softened constraint; pin Cargo to the PR `rev` until merged. Prefer A′ over B **only if** it is genuinely small and generally useful upstream.
- **B — bootstrap-PeerInfo fetch, zero patch:** the controller serves its real (fully public, serde) `peer_info()` bytes at a bootstrap port on the hardcoded DNS; the cell fetches + `register_peer`s it. Always works against unmodified official main; adds one tiny side-channel listener.

The spike proves whichever it lands on end-to-end. **All three are zero-discovery** (no etcd/NATS/velo-discovery backend); all reach the controller from the one operator-hardcoded coordinate.

**Files:**
- Create: `rust/aiperf/examples/velo_cell_spike.rs`
- Modify: `rust/aiperf/Cargo.toml` (add the `velo` optional dep + feature so the example builds)

**Interfaces:**
- Produces: the confirmed body of `resolve_controller_peer(controller_addr: &str) -> anyhow::Result<PeerInfo>` and whether a bootstrap endpoint is needed. Later tasks (3, 5, 8) consume this decision.

- [ ] **Step 1: Add the velo dependency + feature** to `rust/aiperf/Cargo.toml`:

```toml
[dependencies]
velo = { git = "https://github.com/ai-dynamo/velo.git", tag = "v0.5.0", default-features = false, optional = true }

[features]
velo = ["dep:velo"]

[[example]]
name = "velo_cell_spike"
required-features = ["velo"]
```

- [ ] **Step 2: Write the spike** `rust/aiperf/examples/velo_cell_spike.rs`. It builds two `Velo` instances in one process — a "controller" that only knows its own bind address and a "cell" that only knows the controller's address string — and attempts, in order, mechanism A then mechanism B, printing which succeeds.

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Spike: prove a cell can reach a controller knowing only its host:port,
//! with unmodified velo v0.5.0 (no discovery backend).
use std::sync::Arc;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use velo::{Handler, TypedContext, Velo};
use velo::backend::tcp::TcpTransportBuilder;

#[derive(Serialize, Deserialize, Clone)]
struct Register { cell_id: u32, cell_peer: Vec<u8> } // cell_peer = serialized PeerInfo
#[derive(Serialize, Deserialize, Clone)]
struct SpecReply { ok: bool }

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    // --- controller: bind a FIXED loopback port, know only its own addr ---
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let controller_addr = listener.local_addr()?.to_string();
    let controller = Velo::builder()
        .add_transport(Arc::new(TcpTransportBuilder::new().from_listener(listener)?.build()?))
        .build().await?;
    let controller_clone = controller.clone();
    controller.register_handler(
        Handler::typed_unary_async("aiperf.cell.register", move |ctx: TypedContext<Register>| {
            let controller = controller_clone.clone();
            async move {
                // Learn the cell's real PeerInfo from the payload so we can reply/reach it.
                let peer: velo::PeerInfo = rmp_serde::from_slice(&ctx.input.cell_peer)?;
                controller.register_peer(peer)?;
                Ok(SpecReply { ok: true })
            }
        }).build())?;

    // --- cell: know ONLY controller_addr ---
    let cell = Velo::builder()
        .add_transport(Arc::new(TcpTransportBuilder::new()
            .from_listener(std::net::TcpListener::bind("127.0.0.1:0")?)?.build()?))
        .build().await?;

    // Mechanism A: placeholder InstanceId + address-only WorkerAddress.
    // (Requires a PUBLIC way to build a WorkerAddress from host:port. If the
    // needed builder is private, this block won't compile — comment it out and
    // the spike falls through to mechanism B.)
    let mech_a = try_mechanism_a(&cell, &controller_addr).await;
    match mech_a {
        Ok(()) => { println!("SPIKE RESULT: mechanism A (placeholder-address) WORKS"); return Ok(()); }
        Err(e) => println!("mechanism A failed: {e:#}"),
    }

    // Mechanism B: bootstrap — controller's REAL PeerInfo fetched out-of-band.
    // Here we hand it directly (simulating a bootstrap fetch); in production the
    // controller serves peer_info() bytes at controller_addr's bootstrap port.
    let controller_peer_bytes = rmp_serde::to_vec(&controller.peer_info())?;
    let controller_peer: velo::PeerInfo = rmp_serde::from_slice(&controller_peer_bytes)?;
    cell.register_peer(controller_peer.clone())?;
    let reply: SpecReply = cell
        .typed_unary::<SpecReply>("aiperf.cell.register")?
        .payload(&Register { cell_id: 0, cell_peer: rmp_serde::to_vec(&cell.peer_info())? })?
        .instance(controller_peer.instance_id())
        .send().await.context("mechanism B register send")?;
    assert!(reply.ok);
    println!("SPIKE RESULT: mechanism B (bootstrap-peerinfo) WORKS");
    Ok(())
}

async fn try_mechanism_a(cell: &Arc<Velo>, controller_addr: &str) -> Result<()> {
    // Attempt to construct a controller PeerInfo from host:port + a placeholder id.
    // Look in /tmp/velo-main/lib/velo/src/transports/tcp/transport.rs:805 (make_tcp_peer)
    // for the idiom: WorkerAddressBuilder::new().add_entry("tcp", b"tcp://addr").build().
    // If WorkerAddressBuilder is not publicly exported, this cannot be written with
    // the public API — return Err and use mechanism B.
    anyhow::bail!("mechanism A not attempted: confirm public WorkerAddress-from-addr API first")
}
```

- [ ] **Step 3: Investigate whether mechanism A is possible with public API.**

Run:
```bash
grep -rn "pub use.*WorkerAddressBuilder\|pub fn add_entry\|pub mod address\|WorkerAddress::from_encoded" /tmp/velo-main/lib/velo/src/lib.rs /tmp/velo-main/lib/velo-ext/src/**/*.rs
```
Expected: determine if `WorkerAddressBuilder` (or an equivalent public constructor) is reachable from outside velo. If yes, implement `try_mechanism_a` using it (`PeerInfo::new(InstanceId::new_v4(), <addr worker_address>)` + `cell.register_peer` + a `typed_unary` send). If no, leave `try_mechanism_a` as a bail.

- [ ] **Step 4: Run the spike**

Run: `cargo run -p aiperf --features velo --example velo_cell_spike`
Expected: prints `SPIKE RESULT: mechanism A ... WORKS` or `SPIKE RESULT: mechanism B ... WORKS`. Record which in the commit message. **If both fail, STOP and report** — the design's connection assumption is broken and needs revisiting before further tasks.

- [ ] **Step 5: Commit**

```bash
git add rust/aiperf/Cargo.toml rust/aiperf/examples/velo_cell_spike.rs rust/Cargo.lock
git commit --no-verify -m "spike(cellular): prove discovery-free velo cell↔controller reach (mechanism <A|B>)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Decision recorded for later tasks:** if mechanism A/A′ → `resolve_controller_peer` builds the peer from `AIPERF_CELL_CONTROLLER_ADDR` + placeholder id, no bootstrap endpoint (Task 8's bootstrap step is skipped). For A′ also open the upstream velo PR against `ai-dynamo/main` and pin Cargo to its `rev`. If mechanism B → the controller serves `peer_info()` bytes at a bootstrap port (`AIPERF_CELL_CONTROLLER_BOOTSTRAP_ADDR`) and `resolve_controller_peer` fetches+deserializes them (Task 8 builds it).

---

## Task 2: Split the transport seam into a module (no behavior change yet)

**Purpose:** Turn `cellular/transport.rs` into `cellular/transport/mod.rs` holding only the transport-neutral seam, so the velo impl can live beside it feature-gated. The TCP impls stay for now (deleted in Task 4) to keep the tree compiling.

**Files:**
- Create: `rust/aiperf/src/cellular/transport/mod.rs` (from the current `transport.rs`, unchanged content)
- Delete: `rust/aiperf/src/cellular/transport.rs`
- Modify: none of the exports (`cellular/mod.rs` still `pub mod transport;`)

**Interfaces:**
- Produces: `CellClient`, `ControllerTransport`, `CellMessage`, `CellTransportError` unchanged at `crate::cellular::transport::*`.

- [ ] **Step 1:** `git mv rust/aiperf/src/cellular/transport.rs rust/aiperf/src/cellular/transport/mod.rs`

- [ ] **Step 2: Verify it still compiles/tests** (pure move).

Run: `cargo test -p aiperf --lib cellular::transport`
Expected: PASS (same tests as before the move).

- [ ] **Step 3: Commit**

```bash
git add -A rust/aiperf/src/cellular/
git commit --no-verify -m "refactor(cellular): move transport seam into transport/mod.rs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Control DTOs + `connect.rs` (`resolve_controller_peer`, velo builders)

**Files:**
- Modify: `rust/aiperf/src/cellular/transport/mod.rs` (add the DTOs)
- Create: `rust/aiperf/src/cellular/transport/connect.rs`
- Modify: `rust/aiperf/src/cellular/mod.rs` (feature-gated `pub mod`/re-export)

**Interfaces:**
- Produces:
  - `CellRegister { cell_id: u32, cell_peer: Vec<u8> }` (serde) — `cell_peer` is `rmp_serde`-encoded `velo::PeerInfo`.
  - `CellPartitionShip { cell_id: u32, data_handle: u128 }` (serde).
  - `CellAck { ok: bool }` (serde).
  - Handler name consts: `HANDLER_REGISTER = "aiperf.cell.register"`, `HANDLER_HEARTBEAT = "aiperf.cell.heartbeat"`, `HANDLER_PARTITION = "aiperf.cell.partition"`.
  - `#[cfg(feature="velo")] fn resolve_controller_peer(env: &ControllerCoordinate) -> anyhow::Result<velo::PeerInfo>` (impl per Task 1).
  - `#[cfg(feature="velo")] fn build_cell_velo() -> anyhow::Result<Arc<velo::Velo>>` and `build_controller_velo(bind: BindSpec) -> anyhow::Result<Arc<velo::Velo>>`.
  - `enum BindSpec { UdsPath(PathBuf) /* unix */, TcpLoopback, TcpBind(SocketAddr) }` and `struct ControllerCoordinate { addr: String, bootstrap_addr: Option<String> }`.

- [ ] **Step 1: Add the DTOs + handler-name consts** to `transport/mod.rs` (transport-neutral, no `velo` gate — they are plain serde types):

```rust
/// The cell's registration request: its `cell_id` plus its own serialized
/// `velo::PeerInfo` (rmp-encoded) so the controller can reach it back.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellRegister {
    /// Zero-based cell identifier (the barrier key).
    pub cell_id: u32,
    /// `rmp_serde`-encoded `velo::PeerInfo` of the registering cell.
    pub cell_peer: Vec<u8>,
}

/// A cell's records-shard partition shipped by rendezvous handle (u128).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellPartitionShip {
    /// The shipping cell's identifier.
    pub cell_id: u32,
    /// The rendezvous `DataHandle` (u128) of the rmp-encoded partition body.
    pub data_handle: u128,
}

/// Generic acknowledgement reply.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellAck {
    /// Whether the controller accepted the message.
    pub ok: bool,
}

/// velo handler name: cell → controller registration (returns `CellLaunchSpec`).
pub const HANDLER_REGISTER: &str = "aiperf.cell.register";
/// velo handler name: cell → controller heartbeat (fire-and-forget).
pub const HANDLER_HEARTBEAT: &str = "aiperf.cell.heartbeat";
/// velo handler name: cell → controller partition ship (returns `CellAck`).
pub const HANDLER_PARTITION: &str = "aiperf.cell.partition";
```

- [ ] **Step 2: Write `connect.rs`** with the builders + `resolve_controller_peer`. Fill the peer-resolution body per the Task 1 decision. UDS on unix, loopback elsewhere; controller binds fixed for k8s.

```rust
// SPDX headers + //! docs …
#![cfg(feature = "velo")]
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::{Context, Result};
use velo::Velo;
use velo::backend::tcp::TcpTransportBuilder;

/// How a cell reaches the controller (operator-hardcoded).
pub struct ControllerCoordinate {
    /// The controller's velo transport address (`host:port` for k8s TCP).
    pub addr: String,
    /// Optional bootstrap address serving the controller's PeerInfo bytes
    /// (only set when the Task-1 spike chose mechanism B).
    pub bootstrap_addr: Option<String>,
}

/// How the controller/cell binds its velo transport.
pub enum BindSpec {
    /// Unix domain socket at this path (local launcher, unix).
    #[cfg(unix)]
    UdsPath(PathBuf),
    /// TCP on an OS-assigned loopback port (local launcher, non-unix or forced).
    TcpLoopback,
    /// TCP bound to a fixed address (k8s controller at a known port).
    TcpBind(SocketAddr),
}

/// Build a cell's velo instance (ephemeral bind).
pub fn build_cell_velo(bind: BindSpec) -> Result<Arc<Velo>> { build_velo(bind) }
/// Build the controller's velo instance (fixed/known bind).
pub fn build_controller_velo(bind: BindSpec) -> Result<Arc<Velo>> { build_velo(bind) }

fn build_velo(bind: BindSpec) -> Result<Arc<Velo>> {
    // NOTE: build() is async; callers run this on a runtime. Returns Arc<Velo>.
    // Bodies below use the confirmed builder from /tmp/velo-main examples.
    // (Written as an async fn in the real impl; shown sync-shaped for brevity —
    // the implementer makes build_velo async and awaits build().)
    unimplemented!("see step 2 real code: async build via TcpTransportBuilder / UdsTransportBuilder")
}

/// Resolve the controller's `PeerInfo` from the hardcoded coordinate.
/// Mechanism A: construct from `coord.addr` + placeholder id.
/// Mechanism B: fetch serialized PeerInfo bytes from `coord.bootstrap_addr`.
pub async fn resolve_controller_peer(coord: &ControllerCoordinate) -> Result<velo::PeerInfo> {
    // Implemented per Task 1 decision. For mechanism B:
    let bootstrap = coord.bootstrap_addr.as_deref()
        .context("bootstrap_addr required for mechanism B")?;
    let bytes = fetch_bootstrap_peer(bootstrap).await?;
    let peer = rmp_serde::from_slice(&bytes).context("decode controller PeerInfo")?;
    Ok(peer)
}
```

The implementer writes the real async `build_velo` (from the spike's confirmed builder chain) and, if mechanism B, `fetch_bootstrap_peer` (a one-shot TCP/HTTP GET) + the controller-side bootstrap server (added in Task 8). If mechanism A, `resolve_controller_peer` builds the peer inline and `bootstrap_addr` stays `None`.

- [ ] **Step 3: Wire the module** in `cellular/mod.rs`:

```rust
pub mod transport;
#[cfg(feature = "velo")]
pub use transport::{CellRegister, CellPartitionShip, CellAck};
```
and inside `transport/mod.rs`: `#[cfg(feature = "velo")] pub mod velo_transport; #[cfg(feature = "velo")] pub mod connect;`

- [ ] **Step 4: Compile-check both feature states**

Run: `cargo build -p aiperf` (no feature) and `cargo build -p aiperf --features velo`
Expected: both PASS (velo path may still be `unimplemented!()` in `build_velo` — that is fine, it is not called yet).

- [ ] **Step 5: Commit**

```bash
git add rust/aiperf/src/cellular/
git commit --no-verify -m "feat(cellular): control DTOs + velo connect seam (resolve_controller_peer)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `VeloControllerTransport` + `VeloCellClient` behind the seam; delete TCP impls

**Files:**
- Create: `rust/aiperf/src/cellular/transport/velo_transport.rs`
- Modify: `rust/aiperf/src/cellular/transport/mod.rs` (remove `TcpCellClient`/`TcpControllerTransport`/`encode_frame`/`read_frame`/`MAX_FRAME_LEN` + their tests; keep the traits + `CellMessage`)
- Modify: `rust/aiperf/src/cellular/mod.rs` (drop `Tcp*` re-exports; add `Velo*` under `#[cfg(feature="velo")]`)

**Interfaces:**
- Consumes: `CellClient`, `ControllerTransport`, `CellMessage`, `CellRegister`, `CellPartitionShip`, `CellAck`, `HANDLER_*`, `resolve_controller_peer`, `build_*_velo` (Tasks 2–3).
- Produces:
  - `VeloControllerTransport` implementing `ControllerTransport` + `pub fn bind_controller(velo: Arc<Velo>, expected_cells: usize, spec_for: SpecFor) -> Self` where `SpecFor = Box<dyn Fn(u32) -> Option<CellLaunchSpec-bytes> + Send + Sync>` (the controller supplies each cell's spec bytes by `cell_id`). `recv()` yields the next `CellMessage`, `None` when all cells shipped.
  - `VeloCellClient` implementing `CellClient` + `pub async fn register(coord, cell_id) -> Result<Vec<u8>>` (returns the CellLaunchSpec bytes from the reply).

- [ ] **Step 1: Write the failing test** in `velo_transport.rs` (`#[cfg(test)]`): two in-process velo instances, a cell registers and ships a heartbeat + a partition; the controller's `recv()` yields both and the register reply carries the spec bytes.

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cell_registers_ships_heartbeat_and_partition() {
    // controller: build velo (TcpLoopback), bind_controller(expected=1, spec_for=|id| Some(vec![id as u8]))
    // cell: build velo (TcpLoopback), resolve controller peer (mechanism from Task 1),
    //       register(cell_id=0) -> spec bytes == vec![0]; send heartbeat; ship a 1-record partition.
    // controller.recv() twice -> Heartbeat then Partition; then partitions complete.
    // (Full body written by the implementer using the seam.)
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p aiperf --features velo --lib cell_registers_ships`
Expected: FAIL (types not defined).

- [ ] **Step 3: Implement `VeloControllerTransport`.** Register the three handlers; each handler pushes into an internal `tokio::mpsc::Sender<Result<CellMessage>>` that `recv()` drains. The `register` handler: `register_peer` the cell from `CellRegister.cell_peer`, count the `cell_id` (dedup), reply with the spec bytes from `spec_for(cell_id)`. The `partition` handler: `velo.get(data_handle)` → rmp-decode → push `CellMessage::Partition`, reply `CellAck{ok:true}`. The `heartbeat` handler: rmp/JSON-decode → push `CellMessage::Heartbeat`. Model the merged-channel + `recv()` on today's `TcpControllerTransport` (`recv` returns `None` when the sender side closes / all cells shipped).

- [ ] **Step 4: Implement `VeloCellClient`.** `register`: `resolve_controller_peer` → `velo.register_peer` → `typed_unary::<Vec<u8>>(HANDLER_REGISTER).payload(&CellRegister{...}).instance(controller.instance_id()).send()` → return spec bytes. `send(CellMessage)` (the `CellClient` trait method): heartbeat → `am_send(HANDLER_HEARTBEAT)`; partition → `velo.register_data(rmp(partition))` then `typed_unary::<CellAck>(HANDLER_PARTITION).payload(&CellPartitionShip{...})`.

- [ ] **Step 5: Run the test to green**

Run: `cargo test -p aiperf --features velo --lib cell_registers_ships`
Expected: PASS.

- [ ] **Step 6: Delete the TCP impls** from `transport/mod.rs` (`TcpCellClient`, `TcpControllerTransport`, `encode_frame`, `read_frame`, `MAX_FRAME_LEN`, and the two TCP tests). Update `cellular/mod.rs` re-exports: remove `CellClient, ... TcpCellClient, TcpControllerTransport`, keep `CellClient, CellMessage, CellTransportError, ControllerTransport`, add `#[cfg(feature="velo")] pub use transport::velo_transport::{VeloCellClient, VeloControllerTransport};`.

- [ ] **Step 7: Compile both feature states + clippy**

Run: `cargo build -p aiperf` && `cargo build -p aiperf --features velo` && `cargo clippy -p aiperf --features velo --all-targets`
Expected: PASS. (Note: `runner` will now fail to build because it still references `Tcp*` — fixed in Tasks 5–8. If subagent-driven, this task's gate is the `aiperf`-crate build+test; the runner is repaired in its own tasks. If that inter-task red is undesirable, land Tasks 4–8 as one reviewer gate.)

- [ ] **Step 8: Commit**

```bash
git add rust/aiperf/src/cellular/
git commit --no-verify -m "feat(cellular): velo CellClient/ControllerTransport; remove TCP transport

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `CellLauncher` seam + `LocalLauncher`

**Files:**
- Create: `rust/runner/src/cell_launcher.rs`
- Modify: `rust/runner/src/lib.rs` (`pub mod cell_launcher;`)
- Modify: `rust/runner/Cargo.toml` (`velo` feature → `["aiperf/velo"]`)

**Interfaces:**
- Produces:
  - `struct CellLaunchContext { cell_count: u32, controller_addr: String, bootstrap_addr: Option<String> }`
  - `struct CellHandle` wrapping either a `tokio::process::Child` (local) or nothing (k8s), exposing `async fn wait_failure(&mut self) -> Option<String>`.
  - `trait CellLauncher { fn launch(&self, ctx: &CellLaunchContext) -> Result<Vec<CellHandle>>; }`
  - `struct LocalLauncher; struct K8sLauncher;`
  - `fn select_launcher() -> Box<dyn CellLauncher>` reading `AIPERF_CELL_LAUNCHER` (default `local`).
  - Env consts: `CELL_LAUNCHER_ENV = "AIPERF_CELL_LAUNCHER"`.

- [ ] **Step 1: Write the failing test**: `LocalLauncher::launch` with `cell_count=2` spawns two `aiperf --cell` children whose env carries `AIPERF_CELL_ID` ∈ {0,1} and `AIPERF_CELL_CONTROLLER_ADDR == ctx.controller_addr`. (Use a stub exe or assert on the constructed `Command` via a seam; simplest: a unit test asserting `LocalLauncher::cell_command(&ctx, cell_id)` sets the right env — extract command construction into a testable helper.)

```rust
#[test]
fn local_launcher_sets_cell_env() {
    let ctx = CellLaunchContext { cell_count: 2, controller_addr: "uds:/tmp/x.sock".into(), bootstrap_addr: None };
    let cmd = LocalLauncher.cell_command(&ctx, 1);
    let envs: std::collections::HashMap<_,_> = cmd.get_envs()
        .filter_map(|(k,v)| Some((k.to_str()?.to_string(), v?.to_str()?.to_string()))).collect();
    assert_eq!(envs.get("AIPERF_CELL_ID").map(String::as_str), Some("1"));
    assert_eq!(envs.get("AIPERF_CELL_CONTROLLER_ADDR").map(String::as_str), Some("uds:/tmp/x.sock"));
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p aiperf-cli --features velo cell_launcher`
Expected: FAIL (no such module).

- [ ] **Step 3: Implement** `cell_launcher.rs`. `LocalLauncher::cell_command(ctx, cell_id)` builds `Command::new(current_exe()).arg("--cell")` with env `AIPERF_CELL_ID`, `AIPERF_CELL_COUNT`, `AIPERF_CELL_CONTROLLER_ADDR` (and `AIPERF_CELL_CONTROLLER_BOOTSTRAP_ADDR` when `Some`), `stdout(null)`, `stderr(inherit)`, `kill_on_drop(true)`. `launch` spawns `cell_count` of them and returns `CellHandle`s wrapping the children. `K8sLauncher::launch` returns an empty `Vec` (pods already exist; the controller just waits on the barrier) — with a `tracing::info!` naming `cell_count`. `wait_failure` for a local handle awaits `child.wait()` and returns `Some(msg)` on non-zero; for k8s returns `None` (pod failure surfaces via the missing registration + a controller-side timeout added in Task 8).

- [ ] **Step 4: Run to green**

Run: `cargo test -p aiperf-cli --features velo cell_launcher`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/runner/src/cell_launcher.rs rust/runner/src/lib.rs rust/runner/Cargo.toml rust/Cargo.lock
git commit --no-verify -m "feat(runner): CellLauncher seam + LocalLauncher/K8sLauncher

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Cell bootstrap over velo (`cellular_cell.rs` + `main.rs`) — spec via `register`, not stdin

**Files:**
- Modify: `rust/runner/src/cellular_cell.rs` (`CellRecordsShipper` over velo; keep the env-const names)
- Modify: `rust/runner/src/main.rs` (`run_cell` reads env coordinate, builds cell velo, `register`s to fetch the spec, then runs the ordinary execute path)

**Interfaces:**
- Consumes: `VeloCellClient`, `resolve_controller_peer`, `CellLaunchSpec` (unchanged type), the `AIPERF_CELL_*` env consts.
- Produces: `CellRecordsShipper::ship_records(records, epoch_ns)` unchanged signature, now sending over velo; `run_cell()` no longer reads a stdin spec.

- [ ] **Step 1: Write the failing test** (`cellular_cell.rs` `#[cfg(test)]`): a `VeloControllerTransport` stub whose `spec_for` returns a known `CellLaunchSpec`; a `VeloCellClient::register` returns bytes that deserialize to that spec. (This overlaps Task 4's test; here assert the *runner-side* `fetch_spec_from_controller(coord, cell_id) -> CellLaunchSpec` helper end-to-end.)

- [ ] **Step 2: Run to confirm fail** — `cargo test -p aiperf-cli --features velo fetch_spec` → FAIL.

- [ ] **Step 3: Implement.** In `cellular_cell.rs`: `CellRecordsShipper::ship` builds a `VeloCellClient` (via `build_cell_velo` + `resolve_controller_peer`) and sends `CellMessage::Heartbeat` then the partition (rendezvous), replacing the `TcpCellClient::connect`. Add `fetch_spec_from_controller(coord, cell_id) -> Result<CellLaunchSpec>` = register + rmp-decode reply. In `main.rs`: `run_cell` reads `AIPERF_CELL_ID`/`AIPERF_CELL_COUNT`/`AIPERF_CELL_CONTROLLER_ADDR`(+bootstrap) from env, builds a small multi-thread runtime, `fetch_spec_from_controller`, sets the same `AIPERF_CELL_*` env the stdin path set (partition/controller/phase-bases), then runs `run_v2` on the fetched envelope. Remove the stdin `CellLaunchSpec` read.

- [ ] **Step 4: Run to green** — `cargo test -p aiperf-cli --features velo` → PASS (cell tests).

- [ ] **Step 5: Commit**

```bash
git add rust/runner/src/cellular_cell.rs rust/runner/src/main.rs
git commit --no-verify -m "feat(runner): cell fetches CellLaunchSpec + ships records over velo

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Controller rework (`cellular_controller.rs`) — launcher + velo transport + barrier

**Files:**
- Modify: `rust/runner/src/cellular_controller.rs` (`run_cellular` uses `VeloControllerTransport` + `CellLauncher`; the slice/merge/report body unchanged)

**Interfaces:**
- Consumes: `select_launcher`, `CellLaunchContext`, `VeloControllerTransport::bind_controller`, `build_controller_velo`, the unchanged `build_cell_envelope`/`phase_ordinal_bases`/`merge`.
- Produces: `run_cellular` unchanged public signature + `CellularRunOutcome`.

- [ ] **Step 1: Write the failing test** — reuse the existing `cellular_controller` tests (they cover validation/slicing, which are unchanged). Add one test that `bind_controller` with a `spec_for` closure returns the correct per-cell spec bytes for `cell_id`.

- [ ] **Step 2: Run** the existing `cellular_controller` unit tests → they must still PASS (validation/slice logic untouched); the new `spec_for` test FAILs until wired.

- [ ] **Step 3: Implement.** In `run_cellular`: build the controller velo (`build_controller_velo` — `TcpBind` for k8s from the controller's own known addr, or `TcpLoopback`/UDS locally), precompute each cell's `CellLaunchSpec` (the existing `build_cell_envelope` + phase bases, keyed by `cell_id`), `bind_controller(velo, cell_count, spec_for)`. Replace `spawn_cell` loop with `select_launcher().launch(&ctx)`; keep the child-failure watcher (`wait_failure`) feeding the same `select!`-against-failure the current code uses. The partition-collect loop, merge, report write, export plane, and heartbeat sidecar are **unchanged** (they already consume `ControllerTransport::recv()` and `RecordsShardPartition`). Remove `spawn_cell` and its stdin serialization.

- [ ] **Step 4: Run to green** — `cargo test -p aiperf-cli --features velo cellular_controller` → PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/runner/src/cellular_controller.rs
git commit --no-verify -m "feat(runner): controller drives cells over velo via CellLauncher + barrier

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Bootstrap endpoint (only if Task 1 chose mechanism B) + k8s barrier timeout

**Files:**
- Modify: `rust/aiperf/src/cellular/transport/connect.rs` (`serve_bootstrap_peer`, `fetch_bootstrap_peer`)
- Modify: `rust/runner/src/cellular_controller.rs` (start the bootstrap server; add a registration-barrier timeout)

**Interfaces:**
- Produces: `#[cfg(feature="velo")] async fn serve_bootstrap_peer(bind: SocketAddr, peer: PeerInfo) -> Result<JoinHandle<()>>` (a one-shot-per-connection TCP server returning the rmp-encoded PeerInfo); `fetch_bootstrap_peer(addr) -> Result<Vec<u8>>`.

- [ ] **Step 1 (skip entirely if Task 1 = mechanism A).** Write the failing test: `serve_bootstrap_peer` + `fetch_bootstrap_peer` round-trip a `PeerInfo` over loopback.

- [ ] **Step 2: Run to confirm fail.**

- [ ] **Step 3: Implement** a minimal length-prefixed TCP exchange (server writes `u32` len + rmp(PeerInfo) to each accepted connection; client reads it). Wire the controller to start it on `bootstrap_addr` before launching cells, and the cell's `resolve_controller_peer` to fetch from it. Add a **barrier timeout** in `run_cellular` (a `tokio::time` deadline around the partition/registration collect) so a k8s run where a cell pod never registers aborts loudly instead of hanging (the k8s launcher has no child-exit signal). Default e.g. `AIPERF_CELL_REGISTER_TIMEOUT_SECS` (generous; documented).

- [ ] **Step 4: Run to green.**

- [ ] **Step 5: Commit** (`feat(cellular): controller bootstrap peer endpoint + k8s barrier timeout`).

---

## Task 9: Fail-closed without the `velo` feature

**Files:**
- Modify: `rust/runner/src/main.rs` (or `cellular_controller.rs`) — the `cells>1` detection path when built without `velo`.

**Interfaces:**
- Consumes: `cell_count_from_envelope` (unchanged).
- Produces: a `#[cfg(not(feature="velo"))]` arm emitting a typed `success:false` execution envelope: "aiperf runner built without the `velo` feature; multi-cell runs (`cells>1`) require it".

- [ ] **Step 1: Write the failing test** (a `#[cfg(not(feature="velo"))]` unit test): `cells>1` envelope → the controller-detection helper returns the fail-closed diagnostic, not a run.

- [ ] **Step 2: Run to confirm fail** — `cargo test -p aiperf-cli` (no feature) → FAIL.

- [ ] **Step 3: Implement.** In `main.rs`, the `run_controller` branch is `#[cfg(feature="velo")]`; add a `#[cfg(not(feature="velo"))]` sibling that, on `execute` + `cells>1`, calls `emit_cellular_failure(benchmark_id, "velo_feature_required", "...")`. `cells=1` is unaffected (never enters this branch).

- [ ] **Step 4: Run to green** — `cargo test -p aiperf-cli` (no feature) → PASS; `cargo build -p aiperf-cli` (no feature) → PASS.

- [ ] **Step 5: Commit** (`feat(runner): fail closed on cells>1 without the velo feature`).

---

## Task 10: Port `test_cellular.rs` onto the velo LocalLauncher

**Files:**
- Modify: `rust/e2e/tests/test_cellular.rs`
- Modify: `rust/e2e/Cargo.toml` if a `velo` feature passthrough is needed for the e2e crate.

**Interfaces:** Consumes the full `aiperf profile --cells N` path (Python frontend → runner controller → cells over velo).

- [ ] **Step 1:** Add `#![cfg(feature = "velo")]`-gating (or a `required-features`) so the e2e cellular tests build/run only with velo. Confirm the tests launch `aiperf` built with `--features velo` (adjust the harness's cargo invocation / binary selection).

- [ ] **Step 2: Run** `cargo test -p aiperf-e2e-tests --features velo test_cellular_run_from_python_frontend -- --nocphr="" `

Run: `cargo test -p aiperf-e2e-tests --features velo test_cellular`
Expected: PASS — `--cells 3` runs end-to-end over velo and reports the full budget.

- [ ] **Step 3: Run** the byte-parity test.

Run: `cargo test -p aiperf-e2e-tests --features velo test_cellular_matches_single_cell`
Expected: PASS — 3-cell reproduces 1-cell ISL/OSL distributions byte-for-byte (unchanged assertion; now over velo).

- [ ] **Step 4: Commit** (`test(e2e): run cellular e2e over the velo LocalLauncher`).

---

## Task 11: Port `test_graph_cellular.rs` onto the velo LocalLauncher

**Files:**
- Modify: `rust/e2e/tests/test_graph_cellular.rs`

- [ ] **Step 1:** Same feature-gating + runner-build adjustment as Task 10 for the graph path.

- [ ] **Step 2: Run**

Run: `cargo test -p aiperf-e2e-tests --features velo test_graph_cellular`
Expected: PASS — `--cells 3` over a `dag_jsonl` dataset reproduces the 1-cell record count + input-token distribution (unchanged assertion; now over velo).

- [ ] **Step 3: Commit** (`test(e2e): run graph cellular e2e over the velo LocalLauncher`).

---

## Task 12: Docs — agent files, canonical cellular spec, indexes

**Files:**
- Modify: `rust/AGENTS.md`, `rust/CLAUDE.md`, `rust/.github/copilot-instructions.md`, `rust/.cursor/rules/python.mdc` (identical body)
- Modify: `specs/cellular.md` in place so its built cross-host transport section
  states the resulting behavior
- Modify: `specs/README.md` (keep the canonical cellular entry current)
- Modify: `llms.txt` (reference the new transport + feature)

- [ ] **Step 1: Edit `specs/cellular.md` in place** so the built cross-host
  transport section includes these current contracts:

```markdown
A velo-backed implementation (official `ai-dynamo/velo` v0.5.0, no fork) sits
behind the `CellClient`/`ControllerTransport` seam: `aiperf.cell.register`
(typed-unary; returns the `CellLaunchSpec` and ticks the count
barrier), `aiperf.cell.heartbeat` (am_send), `aiperf.cell.partition` (rendezvous handle → the
records-shard partition). Cells reach the controller with **zero discovery** from one
operator-configured DNS:port via
`resolve_controller_peer` (<mechanism chosen by the spike>). A `CellLauncher` seam splits
local-subprocess-over-velo (UDS on unix / TCP-loopback on Windows; the `test_cellular`/
`test_graph_cellular` e2e path) from k8s-pod (no spawn; barrier + registration timeout). All
velo code is behind the `velo` cargo feature: `cells=1` is byte-unchanged without it and
`cells>1` fails closed. The static partition, `IssuanceAuthority`, `RecordsShard`,
`MetricsHeartbeat`, and both merges are unchanged; byte-parity is preserved exactly where it was.
Out of scope: the k8s operator/JobSet/CRD (the Python operator is the reference), dynamic credit
routing, cross-cell sidecar telemetry aggregation.
```

- [ ] **Step 2: Update the four agent files' identical body** — the `aiperf`/`aiperf` crate-table cellular note (velo transport + `velo` feature) and the "Build, test, run" cellular paragraph (add `cargo build -p aiperf-cli --features velo`; note `cells>1` needs the feature). Make the SAME edit in all four.

- [ ] **Step 3: Update `specs/README.md` and `llms.txt`** where they summarize
  cellular transport and crate features.

- [ ] **Step 4: Run the guards**

Run: `source .venv/bin/activate && /usr/bin/python3 tools/check_agent_files_sync.py && /usr/bin/python3 tools/check_docs_current.py`
Expected: both exit 0.

- [ ] **Step 5: Review the documentation diff** and confirm it describes only
  the resulting architecture and requirements.

---

## Task 13: Full-suite green + clippy/fmt

- [ ] **Step 1:** `cargo fmt` ; `cargo clippy -p aiperf -p aiperf-cli --features velo --all-targets -- -D warnings`
- [ ] **Step 2:** `cargo test -p aiperf --features velo --lib` → PASS
- [ ] **Step 3:** `cargo test -p aiperf-cli --features velo` → PASS
- [ ] **Step 4:** `cargo build -p aiperf-cli` (no feature) → PASS ; `cargo test -p aiperf-cli` (no feature) → PASS (fail-closed test)
- [ ] **Step 5:** `cargo test -p aiperf-e2e-tests --features velo test_cellular test_graph_cellular` → PASS
- [ ] **Step 6: Commit** any fmt/clippy fixups (`chore: fmt + clippy for velo cell transport`).

---

## Self-Review

**Spec coverage:**
- §4.1 velo transport behind the seam → Tasks 3, 4. ✅
- §4.2 zero-discovery identity/connection + spike → Tasks 1, 3, 8. ✅
- §4.3 `CellLauncher` (Local/K8s) → Task 5, consumed in Task 7. ✅
- §4.4 `CellLaunchSpec` over `register` (not stdin) → Tasks 4, 6, 7. ✅
- §4.5 feature gate + fail-closed → Tasks 1 (feature), 9 (fail-closed). ✅
- §5 data flow (controller build→launch→barrier→merge) → Task 7. ✅
- §6 failure/lifecycle (child-exit watcher + k8s timeout) → Tasks 5, 7, 8. ✅
- §7 testing (unit + e2e over velo) → Tasks 4, 5, 6, 10, 11. ✅
- §8 docs → Task 12 (canonical spec and index updates). ✅
- Large-partition rendezvous → Task 4 (partition handler). ✅

**Placeholder scan:** The only deferred code is `resolve_controller_peer`/`build_velo` bodies, gated on the Task 1 spike by design (an explicit spike, not a hidden TODO) — Task 3 notes exactly what the implementer writes for each mechanism. No "add error handling"/"write tests"-style placeholders.

**Type consistency:** `CellRegister`/`CellPartitionShip`/`CellAck` and `HANDLER_*` are defined in Task 3 and consumed with the same names/fields in Task 4/6/7. `CellLaunchContext`/`CellHandle`/`CellLauncher` defined in Task 5, consumed in Task 7. `resolve_controller_peer`/`build_*_velo` defined in Task 3, consumed in Tasks 4, 6, 7, 8. `CellLaunchSpec` is the existing type, unchanged.

**Note on inter-task build state:** Tasks 4–8 transiently leave the `runner` crate red (the TCP impls are removed in Task 4 before the runner is rewired in Tasks 6–7). Reviewer gates should treat Tasks 4→8 as a connected sequence (each is independently *reviewable*, but the *runner build* only goes green again at Task 7/9). If a strictly-green-between-tasks policy is required, land 4–9 under one gate.
