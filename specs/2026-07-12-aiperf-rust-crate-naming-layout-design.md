<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Rust crate identity and workspace layout

**Date:** 2026-07-12
**Author:** Anthony Casagrande (Tech Lead) + Codex
**Status:** decided — not implemented
**Decision:** AIPerf-owned Cargo packages use the `aiperf` namespace, their
workspace directories omit the redundant `aiperf-` prefix, and the
cross-product `loadgen-core` substrate keeps its neutral package identity.

This is a greenfield naming decision. Package identities become expensive to
change once downstream manifests, imports, documentation URLs, release
automation, and external consumers depend on them. Filesystem paths are local
implementation details, but choosing their convention now avoids permanent
mixed styles.

## 1. Scope and current state

This spec governs:

- Cargo package names (`[package].name`);
- workspace member directory names below `crates/`;
- the corresponding Rust crate identifiers used in source;
- the criteria for a package to use a namespace other than `aiperf`;
- the intended product-neutral identity of `loadgen-core`.

It does not rename a crate or directory by itself. At the time of this
decision, most AIPerf package names and directories both carry `aiperf-`, for
example `crates/aiperf-clock` containing package `aiperf-clock`.
`loadgen-core` already has the intended neutral name and lives at
`crates/loadgen-core`. The directory shortening in this spec remains unbuilt.

Cargo terminology matters here:

- a **package** is the unit named by `[package].name` and selected by
  `cargo -p`;
- a package contains one or more **crate targets** (library, binary, examples,
  tests, or benches);
- a **workspace directory** is only a repository path and need not equal the
  package name;
- Cargo exposes a hyphenated library package such as `aiperf-clock` to Rust
  source as the identifier `aiperf_clock`.

## 2. Normative naming rules

The words MUST, MUST NOT, SHOULD, and MAY are normative.

### 2.1 AIPerf-owned package identity

1. The umbrella library package MUST be named `aiperf`.
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
   underscore mapping (`aiperf-clock` -> `aiperf_clock`). A package MUST NOT add
   a custom `[lib].name` merely to remove the `aiperf_` prefix from imports.

### 2.2 Workspace directory identity

1. An AIPerf capability package SHOULD live at `crates/<capability>`, not
   `crates/aiperf-<capability>`.
2. The umbrella `aiperf` package SHOULD remain at `crates/aiperf` because it
   has no capability suffix to use as a shorter directory name.
3. Directory names MUST be kebab-case and SHOULD equal the capability suffix
   exactly.
4. Code, scripts, CI, and documentation MUST treat Cargo metadata as the
   authority for package identity. They MUST NOT infer a package name by
   prepending or stripping text from its directory basename.

The path is already scoped by the AIPerf repository and `crates/` directory;
repeating the product prefix there adds visual noise without supplying a new
namespace. The package name remains fully qualified because it is meaningful
outside that path.

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
| Workspace path while hosted here | `crates/loadgen-core` |
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

## 4. Target workspace mapping

The target changes directory paths only. Existing AIPerf Cargo package names
remain stable, and `loadgen-core` remains unchanged.

| Target directory | Cargo package | Rust identifier |
|---|---|---|
| `crates/aiperf` | `aiperf` | `aiperf` |
| `crates/accuracy` | `aiperf-accuracy` | `aiperf_accuracy` |
| `crates/adaptive` | `aiperf-adaptive` | `aiperf_adaptive` |
| `crates/clock` | `aiperf-clock` | `aiperf_clock` |
| `crates/core` | `aiperf-core` | `aiperf_core` |
| `crates/dataset` | `aiperf-dataset` | `aiperf_dataset` |
| `crates/endpoints` | `aiperf-endpoints` | `aiperf_endpoints` |
| `crates/extensions` | `aiperf-extensions` | `aiperf_extensions` |
| `crates/gpu-telemetry` | `aiperf-gpu-telemetry` | `aiperf_gpu_telemetry` |
| `crates/graph` | `aiperf-graph` | `aiperf_graph` |
| `crates/metrics` | `aiperf-metrics` | `aiperf_metrics` |
| `crates/network-latency` | `aiperf-network-latency` | `aiperf_network_latency` |
| `crates/rng` | `aiperf-rng` | `aiperf_rng` |
| `crates/runner` | `aiperf-runner` | `aiperf_runner` |
| `crates/server-metrics` | `aiperf-server-metrics` | `aiperf_server_metrics` |
| `crates/timing` | `aiperf-timing` | `aiperf_timing` |
| `crates/transport-http` | `aiperf-transport-http` | `aiperf_transport_http` |
| `crates/loadgen-core` | `loadgen-core` | `loadgen_core` |

The `crates/core` row records only the mechanical path implied by the naming
rule. It does not ratify `aiperf-core` as the best long-term responsibility
name. The current package combines OpenAI Chat/SSE wire helpers with a
collector-backed observer adapter, while `loadgen-core` owns the lower neutral
contract. That responsibility boundary SHOULD receive a separate semantic
review before `aiperf-core` is treated as permanent; this spec does not rename
or split it.

## 5. Ecosystem precedent and rationale

Rust workspaces use both directory styles; Cargo does not require the package
and directory basenames to match. The durable distinction is:

- package names need global context;
- repository paths already have local context.

Examples reviewed for this decision:

- AI-Dynamo uses short paths such as `lib/runtime`, `lib/llm`, and
  `lib/mocker`, while their packages are `dynamo-runtime`, `dynamo-llm`, and
  `dynamo-mocker`. Its separately identified KV block-manager family uses
  `kvbm-*`. This is the closest organizational precedent for AIPerf.
- Wasmtime uses short paths such as `crates/environ` while retaining package
  identities such as `wasmtime-environ`; it additionally marks unsupported
  internals with `wasmtime-internal-*` package names.
- Tokio and Tracing repeat their namespace in both workspace paths and package
  names (`tokio-util`, `tracing-core`). That style is valid, but AIPerf chooses
  the shorter local path convention used by AI-Dynamo and Wasmtime.

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

`crates/aiperf-clock` containing `aiperf-clock` is valid and is the current
layout, but it is rejected as the greenfield target because the repository path
already supplies the AIPerf scope. Exact path/package equality is not valuable
enough to retain the repeated prefix.

### Prefixing every workspace member with `aiperf-`

This is rejected for `loadgen-core`. Workspace membership does not imply
product ownership, and the package is explicitly intended to cross the
AIPerf/AI-Dynamo boundary.

### Prefixing the shared contract with `dynamo-`

This is rejected for the same reason. AI-Dynamo Mocker is a consumer, not the
sole owner of the shared contract.

## 7. Migration plan

Directory shortening SHOULD land as one mechanical change before more external
automation hard-codes the current paths:

1. Move each `crates/aiperf-<capability>` directory to
   `crates/<capability>` with Git history preserved.
2. Keep every `[package].name` unchanged.
3. Update root workspace dependency paths, relative path dependencies, scripts,
   CI, documentation, tests, and source references.
4. Keep `crates/aiperf` and `crates/loadgen-core` unchanged.
5. Run `cargo metadata --no-deps` and verify the package set and package names
   are identical before and after the move.
6. Run formatting, build, tests, documentation guards, and the agent-file sync
   check required by the repository.

Because package names do not change, normal Rust imports, `cargo -p` selectors,
Cargo.lock package identities, and prospective downstream dependency keys stay
stable. The migration is intentionally a repository-path change, not a public
Rust API rename.

## 8. Conformance gate

The directory migration SHOULD add a small metadata-based repository check.
Given each workspace package's manifest path and package name, it should enforce:

- `aiperf` is at `crates/aiperf`;
- `aiperf-<capability>` is at `crates/<capability>`;
- `loadgen-core` is at `crates/loadgen-core`;
- every additional exception is explicitly allowlisted with a link to its
  ownership decision;
- no check derives Rust identifiers independently of Cargo.

That guard is designed but not built. Until the directory migration lands, it
would correctly fail the current prefixed paths and therefore MUST NOT be
enabled as a required check prematurely.

## 9. Consequences

- Cargo package identities remain explicit and globally recognizable.
- Local paths become shorter and easier to scan.
- AIPerf aligns with AI-Dynamo's separation between short component paths and
  namespaced package identities.
- `loadgen-core` advertises its real cross-product ownership boundary.
- Future extraction of `loadgen-core` does not force consumer import changes.
- Tooling must consult Cargo metadata rather than assume directory/package
  basename equality.
- `aiperf-core` remains a separate responsibility-naming question and is not
  silently blessed by this mechanical convention.

## Addendum — 2026-07-12 (new crates added after mapping table)

Several Cargo packages were added or became product-relevant after the target
mapping table in §4 was written. The naming policy still holds: package names
retain the `aiperf-` prefix, while the future target directory should use the
short local capability name unless explicitly excepted.

Additional target mappings:

| Target directory | Cargo package | Rust identifier |
|---|---|---|
| `crates/content-server` | `aiperf-content-server` | `aiperf_content_server` |
| `crates/mock-rs` | `aiperf-mock-rs` | `aiperf_mock_rs` |
| `crates/prometheus` | `aiperf-prometheus` | `aiperf_prometheus` |
| `crates/telemetry-archive` | `aiperf-telemetry-archive` | `aiperf_telemetry_archive` |
| `crates/transport-grpc` | `aiperf-transport-grpc` | `aiperf_transport_grpc` |

This addendum updates only the inventory. The directory migration, independent
`loadgen-core` versioning, Dynamo consumption, and metadata conformance guard
remain unimplemented.

## Addendum — 2026-07-12 (directory migration + conformance guard implemented)

The §7 directory migration has landed and the §8 conformance gate is now built
and enabled. This addendum is authoritative where it conflicts with the top
`Status:` line and the "remain unimplemented" note above.

What changed:

- Every `crates/aiperf-<capability>` directory was moved to
  `crates/<capability>` with `git mv` (history preserved). `crates/aiperf`
  and `crates/loadgen-core` are unchanged. All 23 `[package].name` values are
  byte-identical before and after: package `aiperf-clock` now lives at
  `crates/clock`, and so on across the §4/§287 mapping tables. `cargo metadata
  --no-deps` reports the identical package set, so `cargo -p` selectors,
  imports, and `Cargo.lock` identities are stable.
- Root `[workspace.dependencies]` path values, sibling `../aiperf-<cap>` path
  dependencies, the excluded test-fixture manifest's `../../../../aiperf-<cap>`
  paths, and the `exclude` entry were repointed to the short directories. The
  `members = ["crates/*"]` glob needed no change.
- Non-spec source/test/doc references to the moved paths were updated:
  in-source first-line path comments, the Python tests that read Rust source by
  path, `llms.txt`, `docs/dev/`, `plans/`, `crates/mock-rs/PORTING.md`
  (repo-local lines only; the external deprecated `aiperf-rs` checkout
  reference is intentionally preserved), and the architecture-atlas content
  evidence paths and route/slug expectations (crate routes remain keyed on the
  unchanged package name, e.g. `/crates/aiperf-clock`).
- The §8 gate is implemented as `tools/check_crate_layout.py`: it reads
  `cargo metadata --no-deps` (never a directory basename) and enforces
  `aiperf` -> `crates/aiperf`, `aiperf-<cap>` -> `crates/<cap>`, the
  `loadgen-core` allowlisted exception, and fail-closed rejection of any other
  un-prefixed package. It is wired into `.pre-commit-config.yaml` as
  `check-crate-layout`. The architecture-atlas additionally enforces
  package/path identity against `cargo metadata` in
  `apps/architecture-atlas/src/domain/integrity.ts`.

Still unbuilt (unchanged by this addendum): independent `loadgen-core`
SemVer/versioning separate from the workspace version, and actual AI-Dynamo
Mocker consumption of `loadgen-core` across a repository boundary. The
`aiperf-core` responsibility-naming question (§4) also remains open; the
directory move to `crates/core` is mechanical and does not bless the name.

Spec bodies above are preserved verbatim per the repository's append-only spec
policy; this addendum is the authoritative status.

## Addendum — 2026-07-13 (workspace directory renamed `crates/` → `rust/`)

The top-level workspace directory was renamed from `crates/` to `rust/`. This
supersedes every `crates/<...>` path in the body above and in the first two
addenda: the umbrella package now lives at `rust/aiperf`, each
`aiperf-<capability>` package at `rust/<capability>`, and the `loadgen-core`
exception at `rust/loadgen-core`. Package *identity* is unchanged (still
`aiperf` / `aiperf-*` / `loadgen-core`); only the workspace *path* moved. The
rename adopts the language-folder convention (`rust/` holding crates directly,
no extra nesting) used by polyglot repos, replacing the Rust-community
`crates/` convention, so the Rust tree sits beside the Python `src/` tree under
a self-describing top-level name.

`tools/check_crate_layout.py` now resolves `crates_root = repo_root / "rust"`
and reports policy-mandated `rust/<capability>` directories; the
`check-docs-current` guard's crate-manifest regex is `^rust/[^/]+/Cargo\.toml$`;
and the `native-runner` / `rust-docs-guard` CI path filters watch `rust/**`.
The §8 identity-vs-path split is otherwise intact. Everything listed as
"still unbuilt" in the prior addendum remains unbuilt.

Spec bodies above are preserved verbatim per the repository's append-only spec
policy; this addendum is the authoritative status where paths conflict.
