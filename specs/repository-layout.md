<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Repository layout and crate identity

## Purpose

Define the Cargo workspace topology, package identity, and the naming rules that
govern any future package. Filesystem paths are local; package names escape the
workspace through dependency graphs, diagnostics, artifact metadata, and
possible publication, so they carry the product namespace.

## Built

### Workspace members

The workspace uses edition 2024 and resolver 3. AIPerf-owned Cargo packages
carry the `aiperf` namespace; their directories drop the redundant `aiperf-`
prefix and sit directly under `rust/`.

| Directory | Cargo package | Rust identifier |
|---|---|---|
| `rust/loadgen-core` | `loadgen-core` | `loadgen_core` |
| `rust/runtime` | `aiperf-runtime` | `aiperf_runtime` |
| `rust/cli` | `aiperf-cli` | `aiperf_cli` |
| `rust/mock-server` | `aiperf-mock-server` | `aiperf_mock_server` |
| `rust/e2e` | `aiperf-e2e-tests` | `aiperf_e2e_tests` |
| `rust/pyext` | `aiperf-pyext` | `_native` |

Direct dependency direction is `aiperf-cli` → `aiperf-runtime` → `loadgen-core`;
`aiperf-mock-server` → `aiperf-runtime`; `aiperf-pyext` → pyo3. The CLI and mock
server are independent executables.

### Crate responsibilities

- `loadgen-core`: the transport-neutral dispatch/observation contract
  (`Dispatchable`, `RequestSink<R>`, `RequestObserver`, `ObservedUsage`,
  `TraceCollector`, `CollectorObserver`). No engine, router, KV, HTTP, or gRPC
  dependency.
- `aiperf-runtime`: library-only runtime composition. Its capability
  responsibilities are modules addressed as `aiperf_runtime::<module>::`
  (`clock`, `transport`, `endpoints`, `dataset`, `graph`, `metrics_core`,
  `export`, `cellular`, `engine` behind the `engine` feature, and the rest). See
  `docs/module-organization.md` for the full module table.
- `aiperf-cli`: library plus the `aiperf` binary. It owns command routing,
  Config v2 loading and expansion, profile projection, self-execution, cellular
  roles, native searches and sweeps, result rendering, and process signals.
- `aiperf-mock-server`: standalone HTTP/gRPC inference target.
- `aiperf-pyext`: packaging-only pyo3 `cdylib` maturin compiles into
  `aiperf._native`. `make wheel` repacks the `aiperf` binary into the wheel's
  scripts directory through `tools/wheel_repack.py`.
- `aiperf-e2e-tests`: product integration harness.

### `loadgen-core` neutral identity

`loadgen-core` is the intentional naming exception: a transport-neutral contract
intended for both AIPerf and a second consumer (AI-Dynamo Mocker). It depends on
no `aiperf-*` or `dynamo-*` product crate and no backend. Its neutral name
permits future repository extraction without changing the Cargo package name or
Rust imports.

### Naming rules

The MUST/SHOULD words are normative for any new package:

1. The umbrella runtime library is `aiperf-runtime` at `rust/runtime`.
2. Every other AIPerf-owned package is `aiperf-<capability>` (kebab-case) at
   `rust/<capability>`, where `<capability>` names a stable responsibility, not a
   language, backend, or phase.
3. AIPerf-owned packages do not claim bare generic names (`clock`, `metrics`,
   `transport`, `runner`).
4. Rust identifiers use Cargo's hyphen-to-underscore mapping; a package does not
   add a custom `[lib].name` to remove the `aiperf_` prefix. The required
   exception is `aiperf-pyext`: its `[lib].name = "_native"` makes maturin emit
   the configured `aiperf._native` extension module.
5. A package may omit the `aiperf-` prefix only when it is an independently owned
   subsystem (product-neutral API, no product-crate dependency, independent
   versioning) — currently only `loadgen-core`.
6. Code, scripts, CI, and documentation treat Cargo metadata as the authority for
   package identity, never a directory basename.

The metadata-based conformance gate is `tools/check_crate_layout.py`, wired into
`.pre-commit-config.yaml` as `check-crate-layout`. It resolves its root as
`repo_root / "rust"`, enforces the directory/package rules, and fails closed on
any un-prefixed package outside the `loadgen-core` allowlist.

## Future requirements

- `loadgen-core` needs an independent SemVer/versioning policy separate from the
  shared workspace version.
- Actual AI-Dynamo Mocker consumption of `loadgen-core` across a repository
  boundary is not yet wired; a shared-release mechanism and cross-repository
  compatibility tests must precede it.

## Source anchors

- `Cargo.toml` (`[workspace].members`, `[workspace.dependencies]`).
- Each member's `Cargo.toml`.
- `tools/check_crate_layout.py`; `.pre-commit-config.yaml`.
- `docs/module-organization.md`.
