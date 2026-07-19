<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `loadgen-core` merge into `aiperf-runtime`

## Purpose

Retire `loadgen-core` as a standalone Cargo workspace member and fold its
contents into `aiperf-runtime` as a module, eliminating the compiler-enforced
transport-neutrality boundary in favor of a module-level convention. This
reverses the crate's original design intent (documented in
`repository-layout.md`'s now-removed "Future requirements" section) of keeping
it product-neutral for eventual extraction and reuse by AI-Dynamo Mocker across
a repository boundary; that reuse was never wired up, and the plan is retired
as part of this change.

## Built

- `rust/loadgen-core/src/{sink,collector,observer}.rs` move to
  `rust/runtime/src/dispatch/{sink,collector,observer}.rs`, exposed as
  `pub mod dispatch;` in `rust/runtime/src/lib.rs` with `sink`, `collector`,
  `observer` as its submodules. `dispatch/mod.rs` carries the former crate-level
  doc comment, rephrased to describe a module-level (not crate-level)
  transport-neutrality convention.
- All in-tree `use loadgen_core::...` references (25 files under
  `rust/runtime/src/`) become `crate::dispatch::...`.
- `loadgen-core` is removed from `rust/Cargo.toml` workspace `members` and
  `[workspace.dependencies]`; `loadgen-core.workspace = true` is removed from
  `rust/runtime/Cargo.toml` (loadgen-core's own dependencies — `anyhow`,
  `async-trait`, `rustc-hash`, `serde`, `uuid`, plus dev-deps `serde_json`,
  `tokio` — are confirmed present in `aiperf-runtime`'s own manifest, adding any
  that are missing). `rust/loadgen-core/` is deleted.
- Design records and agent docs (`specs/repository-layout.md`,
  `specs/architecture.md`, `specs/websocket-transport.md`, the four synced
  agent files, `llms.txt`) are updated in the same change to reflect five
  workspace crates instead of six, drop the `loadgen-core` naming exception and
  its future-extraction plan, and describe the dispatch seam as
  `aiperf_runtime::dispatch` rather than `loadgen_core`.
- `tools/check_crate_layout.py`'s `loadgen-core` allowlist is reviewed and
  updated if it still special-cases the now-removed crate.

No runtime behavior changes; this is a pure code-organization move. Verified by
`cargo build`, `cargo test -p aiperf-runtime`, `cargo clippy --all-targets`,
`cargo fmt --check`, `tools/check_agent_files_sync.py`, and
`tools/check_docs_current.py`.

## Source anchors

- `rust/runtime/src/dispatch/` (new module, post-move).
- `rust/runtime/src/lib.rs` (`pub mod dispatch;`).
- `rust/Cargo.toml`, `rust/runtime/Cargo.toml` (workspace membership and deps).
- `specs/repository-layout.md` (crate topology and naming rules).
