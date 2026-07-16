<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Rust crate identity and workspace layout

**Date:** 2026-07-12
**Author:** Anthony Casagrande (Tech Lead) + Codex
**Status:** decided; implemented — workspace directory renamed `crates/` → `rust/`, the former per-capability `aiperf-*` library crates are consolidated into modules of `aiperf-runtime`, and the §8 conformance gate is built.
**Decision:** AIPerf-owned Cargo packages use the `aiperf` namespace, their
workspace directories omit the redundant `aiperf-` prefix and live directly
under `rust/`, and the cross-product `loadgen-core` substrate keeps its neutral
package identity.

Package identities become expensive to change once downstream manifests,
imports, documentation URLs, release automation, and external consumers depend
on them. Filesystem paths are local implementation details, but a single
convention avoids permanent mixed styles.

## 1. Scope and current state

This spec governs:

- Cargo package names (`[package].name`);
- workspace member directory names below `rust/`;
- the corresponding Rust crate identifiers used in source;
- the criteria for a package to use a namespace other than `aiperf`;
- the intended product-neutral identity of `loadgen-core`.

The workspace currently holds four product/library packages plus a test-harness
package. Package names carry the product prefix (`aiperf-runtime`, `aiperf-cli`,
`aiperf-mock-server`) while their directories drop it (`rust/runtime`, `rust/cli`,
`rust/mock-server`). `loadgen-core` keeps its neutral name at `rust/loadgen-core`.
The historical per-capability library crates (`aiperf-clock`, `aiperf-metrics`,
`aiperf-transport-http`, and the rest) no longer exist as separate packages:
they are consolidated into modules of the single `aiperf-runtime` crate (see §4).

Cargo terminology matters here:

- a **package** is the unit named by `[package].name` and selected by
  `cargo -p`;
- a package contains one or more **crate targets** (library, binary, examples,
  tests, or benches);
- a **workspace directory** is only a repository path and need not equal the
  package name;
- Cargo exposes a hyphenated library package such as `aiperf-cli` to Rust
  source as the identifier `aiperf_cli`.

## 2. Normative naming rules

The words MUST, MUST NOT, SHOULD, and MAY are normative.

### 2.1 AIPerf-owned package identity

1. The umbrella runtime library package MUST be named `aiperf-runtime`.
2. Every other AIPerf-owned package MUST be named
   `aiperf-<capability>` in kebab-case.
3. `<capability>` MUST describe the package's stable responsibility, not its
   implementation language, current backend, or temporary project phase.
4. AIPerf-owned packages MUST NOT claim bare generic names such as `clock`,
   `metrics`, `transport`, or `runner`. Cargo package identity escapes the
   workspace through dependency graphs, diagnostics, documentation, artifact
   metadata, and possible publication; it therefore needs the product
   namespace.
5. The corresponding Rust identifier MUST use Cargo's normal hyphen-to-
   underscore mapping (`aiperf-cli` -> `aiperf_cli`). A package MUST NOT
   add a custom `[lib].name` merely to remove the `aiperf_` prefix from imports.

### 2.2 Workspace directory identity

1. An AIPerf capability package SHOULD live at `rust/<capability>`, not
   `rust/aiperf-<capability>`.
2. The umbrella `aiperf-runtime` package SHOULD live at `rust/runtime`,
   dropping the `aiperf-` prefix like every other capability package.
3. Directory names MUST be kebab-case and SHOULD equal the capability suffix
   exactly (`aiperf-cli` -> `rust/cli`, `aiperf-mock-server` -> `rust/mock-server`).
4. Code, scripts, CI, and documentation MUST treat Cargo metadata as the
   authority for package identity. They MUST NOT infer a package name by
   prepending or stripping text from its directory basename.

The path is already scoped by the AIPerf repository and the `rust/` language
folder; repeating the product prefix there adds visual noise without supplying a
new namespace. The package name remains fully qualified because it is meaningful
outside that path. `rust/` holds the crates directly, with no extra nesting, so
the Rust tree sits beside the Python `src/` tree under a self-describing
top-level name — the language-folder convention used by polyglot repos, in place
of the Rust-community `crates/` convention.

### 2.3 Independently owned subsystem exception

A workspace package MAY omit `aiperf-` only when it is an independently owned
subsystem rather than an AIPerf implementation detail. Such a package MUST:

- have, or have an approved plan for, at least two product consumers;
- expose a product-neutral API and vocabulary;
- depend on neither `aiperf-*` nor `dynamo-*` product crates;
- own an independent compatibility and versioning policy;
- be usable without an AIPerf CLI, runner, configuration, transport, or report;
- use its own non-generic namespace consistently if it grows into multiple
  packages.

An exception is architectural, not aesthetic. A leaf does not qualify merely
because it currently has few dependencies.

## 3. `loadgen-core` is the intentional exception

`loadgen-core` is the shared transport-neutral dispatch and observation
contract intended for both AIPerf and AI-Dynamo's Mocker. It therefore MUST
remain:

| Concern | Decision |
|---|---|
| Package name | `loadgen-core` |
| Workspace path while hosted here | `rust/loadgen-core` |
| Rust identifier | `loadgen_core` |
| Product dependencies | none: no `aiperf-*` or `dynamo-*` |
| Backend dependencies | none: no HTTP client, engine, router, KV manager, or simulator |
| Consumers | AIPerf now; AI-Dynamo Mocker is the planned second consumer |
| Versioning target | independent SemVer compatibility from AIPerf product releases |

The current implementation already satisfies the dependency-neutral direction:
it owns `Dispatchable`, `RequestSink<R>`, `RequestObserver`, and the lean trace
collector without an AIPerf, Dynamo, HTTP, engine, router, KV, or simulation
dependency. Direct AI-Dynamo consumption and independent versioning are
designed but not built; today its manifest still inherits the workspace
version.

Neither `aiperf-loadgen-core` nor `dynamo-loadgen-core` is acceptable. Either
name would assign a shared contract to one consumer and make the other look
like an adapter to a foreign product. Keeping `loadgen-core` also permits a
future repository extraction without changing the Cargo package name or Rust
imports.

Sharing creates release obligations. Before AI-Dynamo consumes it across a
repository boundary, `loadgen-core` MUST have:

- an explicit supported API surface and SemVer policy;
- a release/source mechanism consumable by both repositories;
- compatibility tests exercised by AIPerf and AI-Dynamo Mocker;
- product-neutral package metadata and documentation;
- a clear ownership path for coordinated breaking changes.

Physical residence in the AIPerf repository does not grant AIPerf semantic
ownership of the package.

## 4. Workspace mapping

The workspace members and their directories are:

| Directory | Cargo package | Rust identifier |
|---|---|---|
| `rust/runtime` | `aiperf-runtime` | `aiperf_runtime` |
| `rust/cli` | `aiperf-cli` | `aiperf_cli` |
| `rust/mock-server` | `aiperf-mock-server` | `aiperf_mock_server` |
| `rust/e2e` | `aiperf-e2e-tests` | `aiperf_e2e_tests` |
| `rust/pyext` | `aiperf-pyext` | `aiperf_pyext` |
| `rust/loadgen-core` | `loadgen-core` | `loadgen_core` |

Every package name follows §2.1; every directory follows §2.2 (the `aiperf-`
prefix is stripped from the path, kept in the package identity). `rust/pyext`
holds the packaging-only pyo3 `cdylib` (`aiperf-pyext`) that maturin compiles
into `aiperf._native`; the workspace `pyproject.toml` at the repository root
drives that build, and the single unified `aiperf` binary (crate `aiperf-cli`)
is repacked into the wheel's scripts directory by `tools/wheel_repack.py`
(via `make wheel`). The published wheel is online-only by default because the
`dynosim` feature it can carry needs an external checkout absent in CI.

### 4.1 Consolidation of the former per-capability crates

The naming rules above were authored when AIPerf carried many separate
`aiperf-<capability>` library crates (`aiperf-clock`, `aiperf-core`,
`aiperf-dataset`, `aiperf-endpoints`, `aiperf-metrics`, `aiperf-timing`,
`aiperf-transport-http`, `aiperf-transport-grpc`, `aiperf-graph`, `aiperf-rng`,
`aiperf-adaptive`, `aiperf-accuracy`, `aiperf-gpu-telemetry`,
`aiperf-network-latency`, `aiperf-server-metrics`, `aiperf-content-server`, and
the rest). Those sixteen library crates are now consolidated into modules of the
single `aiperf-runtime` crate, addressed as `aiperf_runtime::<module>::` (for example
`aiperf_runtime::clock`, `aiperf_runtime::metrics_core`, `aiperf_runtime::transport_http`). No separate
`aiperf-<capability>` library package remains. The module namespaces inherit the
same capability vocabulary the crate names used, so the naming discipline still
governs how a responsibility is named — it now names a module rather than a
package. See [`docs/module-organization.md`](../docs/module-organization.md) for
the full module table.

The naming rules in §2 remain the standing policy for any *future* package: if a
capability is ever re-extracted into its own Cargo package, it MUST be named
`aiperf-<capability>` and live at `rust/<capability>`.

The `aiperf-core` responsibility name was never ratified as a long-term
boundary; the OpenAI Chat/SSE wire helpers and the collector-backed observer
adapter it once held now live inside `aiperf-runtime` alongside the neutral
`loadgen-core` contract. That responsibility boundary is a separate semantic
question this spec does not decide.

## 5. Ecosystem precedent and rationale

Rust workspaces use both directory styles; Cargo does not require the package
and directory basenames to match. The durable distinction is:

- package names need global context;
- repository paths already have local context.

Examples reviewed for this decision:

- AI-Dynamo uses short paths such as `lib/runtime`, `lib/llm`, and
  `lib/mocker`, while their packages are `dynamo-runtime`, `dynamo-llm`, and
  `dynamo-mocker`. Its separately identified KV block-manager family uses
  `kvbm-*`. This is the closest organizational precedent for AIPerf, including
  the maturin bindings layout used for the `aiperf-pyext` packaging `cdylib`
  (`rust/pyext`).
- Wasmtime uses short paths such as `crates/environ` while retaining package
  identities such as `wasmtime-environ`; it additionally marks unsupported
  internals with `wasmtime-internal-*` package names.
- Tokio and Tracing repeat their namespace in both workspace paths and package
  names (`tokio-util`, `tracing-core`). That style is valid, but AIPerf chooses
  the shorter local path convention (package identity fully qualified, directory
  prefix dropped) used by AI-Dynamo and Wasmtime.

Primary references, reviewed 2026-07-12:

- [Cargo workspace members and `crates/*`](https://doc.rust-lang.org/cargo/reference/workspaces.html#the-members-and-exclude-fields)
- [Cargo package `name`](https://doc.rust-lang.org/cargo/reference/manifest.html#the-name-field)
- [Cargo's kebab-case package lint](https://doc.rust-lang.org/cargo/reference/lints.html#non_kebab_case_packages)
- [AI-Dynamo workspace](https://github.com/ai-dynamo/dynamo/blob/main/Cargo.toml)
- [AI-Dynamo `dynamo-runtime`](https://github.com/ai-dynamo/dynamo/blob/main/lib/runtime/Cargo.toml)
- [Tokio workspace](https://github.com/tokio-rs/tokio/blob/master/Cargo.toml)
- [Tracing workspace](https://github.com/tokio-rs/tracing/blob/master/Cargo.toml)
- [Wasmtime workspace](https://github.com/bytecodealliance/wasmtime/blob/main/Cargo.toml)

These repositories are precedent, not authority. The normative rules are in
this spec.

## 6. Rejected alternatives

### Bare AIPerf package names

Names such as `clock`, `metrics`, or `transport-http` are rejected. They lose
ownership in `cargo tree`, diagnostics, documentation, artifact metadata, and
downstream manifests, and they compete with broad ecosystem vocabulary.

### Prefixing both the directory and package

`rust/aiperf-cli` containing `aiperf-cli` is valid but rejected because the
repository path already supplies the AIPerf scope. Exact path/package equality
is not valuable enough to retain the repeated prefix.

### Prefixing every workspace member with `aiperf-`

This is rejected for `loadgen-core`. Workspace membership does not imply
product ownership, and the package is explicitly intended to cross the
AIPerf/AI-Dynamo boundary.

### Prefixing the shared contract with `dynamo-`

This is rejected for the same reason. AI-Dynamo Mocker is a consumer, not the
sole owner of the shared contract.

## 7. Realized layout

The layout is in place; the migration was a repository-path change, not a public
Rust API rename:

1. Each `aiperf-<capability>` directory dropped the `aiperf-` prefix, with Git
   history preserved; `rust/runtime` and `rust/loadgen-core` carry no capability prefix to strip.
2. Every `[package].name` is unchanged.
3. Root `[workspace.dependencies]` path values, relative path dependencies,
   scripts, CI, documentation, tests, and source references point at the short
   directories; the `members = ["rust/*"]`-style globs resolve them.
4. The top-level workspace directory itself was renamed `crates/` -> `rust/`;
   package *identity* is unchanged, only the workspace *path* moved.
5. `cargo metadata --no-deps` reports an identical package set, so normal Rust
   imports, `cargo -p` selectors, and `Cargo.lock` package identities stay
   stable.

## 8. Conformance gate

The metadata-based repository check is built as `tools/check_crate_layout.py`
and wired into `.pre-commit-config.yaml` as `check-crate-layout`. Given each
workspace package's manifest path and package name from `cargo metadata
--no-deps` (never a directory basename), it enforces:

- `aiperf-runtime` is at `rust/runtime`;
- `aiperf-<capability>` is at `rust/<capability>`;
- `loadgen-core` is the allowlisted exception at `rust/loadgen-core`;
- every additional exception is explicitly allowlisted with a link to its
  ownership decision;
- fail-closed rejection of any other un-prefixed package.

The check resolves its root as `repo_root / "rust"`. The `check-docs-current`
guard's crate-manifest regex is `^rust/[^/]+/Cargo\.toml$`, and the CI path
filters watch `rust/**`. The architecture-atlas additionally enforces
package/path identity against `cargo metadata` in
`apps/architecture-atlas/src/domain/integrity.ts`.

## 9. Consequences

- Cargo package identities remain explicit and globally recognizable.
- Local paths are short and easy to scan under a self-describing `rust/` folder.
- AIPerf aligns with AI-Dynamo's separation between short component paths and
  namespaced package identities.
- `loadgen-core` advertises its real cross-product ownership boundary.
- Future extraction of `loadgen-core` does not force consumer import changes.
- Tooling consults Cargo metadata rather than assuming directory/package
  basename equality.

## 10. Still unbuilt

Two designed elements remain unbuilt:

- an independent `loadgen-core` SemVer/versioning policy separate from the
  shared workspace version;
- actual AI-Dynamo Mocker consumption of `loadgen-core` across a repository
  boundary.

Until those land, `loadgen-core`'s neutral identity is a forward-looking
commitment realized in dependency direction and package naming, not yet in an
external release contract.
