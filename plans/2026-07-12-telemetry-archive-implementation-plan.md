<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Telemetry archive/watch implementation ledger

Authority: `specs/2026-07-11-aiperf-rust-telemetry-archive-watch-design.md` at
`88c5d99f7`. The spec is implementation intent; code and executable receipts are
the completion evidence.

## Completion rule

The feature is complete only when every §21 criterion and every applicable §18
gate has direct evidence, and a separate non-authoring agent has audited the
full spec against the final code and receipts with no surviving finding. A
passing crate test, a product happy path, or absence of TODOs cannot stand in
for that audit.

The final validator is `/root/review_architecture` (Jason). It is intentionally
not assigned implementation work.

## Baseline audit — 2026-07-12

- No `aiperf-prometheus` crate exists.
- No `aiperf-telemetry-archive` crate exists.
- No `TelemetryArchive`, `telemetry_watch`, `FrameIdentityV1`,
  `ArchiveObjectStore`, or `RawEnvelopeProfile` symbol exists under `crates/`.
- `aiperf-runner` already owns the sole product executable, strict runner-v2
  registry/protocol, typed server/GPU/network sidecars, native report assembly,
  and current scheduled phase integration. These are integration substrates,
  not archive implementation.
- The Python package already owns human CLI/config/orchestration and runner-v2
  wire projection. `aiperf watch` is absent.
- The worktree contains unrelated concurrent changes. Every archive commit must
  use path-limited staging/commit and must not absorb or revert those changes.

## Dependency order and ownership

### A. Lossless exposition foundation — Ampere

- [ ] Add IO-free `crates/aiperf-prometheus`.
- [ ] Implement object-safe `ExpositionParser` and strict format selection.
- [ ] Preserve exact metadata, emitted sample names/roles, escaped labels,
  number/timestamp lexemes, exemplars, and source order.
- [ ] Build typed MetricPoints for every accepted semantic role, including
  metadata-only/empty families, repeated points, merged text Info identity,
  component timestamps, Created values, and cumulative histograms.
- [ ] Enforce all parser/cardinality/body bounds atomically.
- [ ] Keep native-compatibility projection separate from strict parsing.
- [ ] Add §18.1 fixtures and the parser-facing Tachometer regressions.
- Evidence: crate tests, fixture corpus, public API docs, commit IDs.

### B. Archive byte/durability authority — Descartes

- [ ] Add domain-neutral `crates/aiperf-telemetry-archive` without runner,
  metrics, GPU, or backend dependencies.
- [ ] Freeze canonical schema/descriptor/digest/JSON/row evidence APIs and
  `FrameIdentityV1`.
- [ ] Implement canonical index keys and permutation-independent persistent
  mutation semantics.
- [ ] Implement exact WAL frame bytes, CRC-32C, BLAKE3, prefix/footer, segment
  sealing, fsync ordering, and qualified-spool locking.
- [ ] Implement create-only genesis, immutable generations, current/preceding
  local heads, recovery, zero/nonzero projection coverage, and lagged WAL
  retirement.
- [ ] Implement receipt observer epochs, immutable target/event batches,
  single-segment range targets, receipt index/head, and recovery observations.
- [ ] Implement raw-envelope profile/nonce reservation/key counts and
  owner-terminal failure semantics.
- [ ] Implement object-store capability/CAS interfaces and deterministic
  in-memory/fault adapters before provider adapters.
- Evidence: byte goldens, crash matrix, corruption/recovery tests, commit IDs.

### C. Physical archive/query implementation — root

- [ ] Integrate the parser model into exact attempt/family/sample/marker/loss/
  raw-reference Arrow descriptors.
- [ ] Implement whole-frame Parquet projection/rotation, zero-row coverage,
  immutable partition descriptors, and deterministic physical sort.
- [ ] Implement head-root query discovery and bounded source/time pruning.
- [ ] Add pinned Arrow/Parquet/DuckDB/Polars/pyarrow compatibility fixtures.
- [ ] Add transactional compaction with logical multiset equality and canonical
  index replacement.
- Evidence: golden archives, independent reader tests, compaction/crash tests.

### D. Source runtime and product integration — root

- [ ] Read every complete runner/Python source file touched by the port before
  editing, including callers, DTOs, reports, phase observers, and tests.
- [ ] Add strict writer/store-access/source/recovery/policy factory registries.
- [ ] Implement profile-bound `ControlPlaneHttpProvider`, exact all-outcome
  fetch envelope, independent per-source fixed deadlines, one in-flight request,
  active cancellation/redeadline, and bounded two-stage decode/projection.
- [ ] Implement the sole archive owner, per-source FIFO epoch strands, global
  reorder, fixed-memory loss/saturation ledger, Clock-stamp bridges, admission,
  ordered stop/finalization, and report health.
- [ ] Add strict `telemetry_watch::{collect,finalize_remote}` runner-v2 DTOs,
  workload/resource requirements, registry capability, stdout terminal outcome,
  and source-free sync preparation.
- [ ] Add Python config/wire/CLI `aiperf watch`; do not add a Rust binary.
- [ ] Add typed native-v2 archive provenance and diagnostic-only required
  failure behavior.
- Evidence: unit, integration, subprocess, signal, resume, and failure receipts.

### E. Attached benchmark integration — root

- [ ] Replace phase-owned telemetry cadence with one run-owned driver per
  physical source without changing native formulas.
- [ ] Implement source-cardinal atomic boundary plans, continuous phase
  membership, exact attempt-or-loss joins, and `PhaseObserver` markers.
- [ ] Deliver native projection synchronously before nonblocking archive
  admission; never backpressure the request path.
- [ ] Prove same-event-stream `NativeMeasurementParityV1`, real paired
  statistical limits, no extra scrapes, and required/best-effort report behavior.
- Evidence: deterministic parity fixtures and real scheduled subprocess runs.

### F. Remote durability, operations, and qualification — root

- [ ] Implement immutable local/filesystem and object-store adapters, strong
  readback verification, visibility horizon, active writer claims, exact CAS,
  ancestry reconciliation, uncertain outcomes, and absent-claim compaction.
- [ ] Implement create-new, exact-resume, and source-free finalize-remote.
- [ ] Add outage/visibility/conflict emulators and durable publication receipts.
- [ ] Add key rotation, orphan inspection, GC, recovery, and query docs.
- [ ] Check in populated standalone and attached `AcceptanceProfileV1` results
  satisfying every §17 numeric gate on qualified hardware.
- Evidence: emulator matrix, operational subprocess receipts, profile artifacts.

## Mandatory final evidence matrix

- [ ] `cargo fmt --check` for all new/changed Rust.
- [ ] `cargo clippy --all-targets` with no new warnings.
- [ ] Full workspace `cargo test` plus focused new-crate/property/crash suites.
- [ ] Python unit/config/wire/CLI tests.
- [ ] Real Python → packaged `aiperf-runner` collect and sync-only subprocesses.
- [ ] Attached scheduled benchmark against the real mock with parity receipts.
- [ ] HTTP 500, slow source, oversized source, malformed exposition, signal,
  forced crash, ENOSPC/inode, object-store outage/lag/conflict, and resume matrix.
- [ ] Pinned cross-reader golden archive queries.
- [ ] Five Tachometer regression fixtures.
- [ ] Standalone and attached performance profiles with measured numeric bounds.
- [ ] Clean path-scoped status for every implementation-owned file.
- [ ] Jason's full requirement-by-requirement audit reports complete passing
  evidence and no unresolved P0/P1/P2/spec-compliance finding.

## Commit discipline

Commit logical increments as they become independently testable. Use
`git commit --only <owned paths>` and preserve all unrelated dirty work. Never
advertise the runner capability until its exact product subprocess and profile
gates pass.
