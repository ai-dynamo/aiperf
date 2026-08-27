<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Rust plugin compilation and rebuild boundaries

## Purpose

This document explains which AIPerf plugin-facing crates are compiled into the
host and plugin libraries, which artifacts remain dynamically shared at
runtime, and exactly which artifacts must be rebuilt after each class of
change. It is a companion to the normative
[`2026-08-26-native-rust-runtime-plugins-design.md`](2026-08-26-native-rust-runtime-plugins-design.md).
If the two documents conflict, the normative design controls.

The central rule is:

> AIPerf API, core, and category SDK crates are precompiled Rust `.rlib`
> artifacts that are statically linked into each consuming executable or
> plugin `cdylib`. They are not separately loaded runtime `.so`, `.dylib`, or
> `.dll` files. The allocator provider is the deliberate shared-library
> exception.

## Installed artifact shape

On Linux, an installation has the following conceptual shape:

```text
AIPerf installation
|-- aiperf
|   |-- statically linked aiperf-plugin-api Rust code
|   |-- statically linked aiperf-core Rust code
|   `-- dynamic import of libaiperf_alloc_v1.so
|-- plugins/
|   |-- libaiperf_export_basic.so
|   |   |-- statically linked API/core/export-SDK Rust code
|   |   `-- dynamic import of libaiperf_alloc_v1.so
|   |-- libaiperf_export_parquet.so
|   |-- libaiperf_export_otlp.so
|   |-- libaiperf_export_mlflow.so
|   |-- libaiperf_export_wandb.so
|   |-- libaiperf_transport_http.so
|   |-- libaiperf_transport_grpc.so
|   `-- other endpoint and transport packages
`-- libaiperf_alloc_v1.so
```

Windows uses `.dll` artifacts and macOS uses `.dylib` artifacts. Each plugin's
main artifact is a Rust `cdylib`. The SDK/core `.rlib` artifacts are link inputs
to those `cdylib` files; they do not form another layer of runtime plugins.

A plugin can also have private distribution-controlled native dependencies in
its declared executable artifact closure. Those dependencies are separate
runtime libraries only when the plugin's implementation dependency requires
that native topology. They are not AIPerf API/core/SDK libraries, and they are
acquired into an owned immutable snapshot, hashed, closure-checked, and bound in
the installation lock as part of that plugin package. Generation 1 does not
authenticate a third-party author's identity or signature: local installation
authority is the trust root. Only first-party distribution inventory has the
additional authenticated-distribution guarantee.

## Crate responsibilities

### `aiperf-plugin-api`

This crate owns the native Rust host/plugin boundary:

- `AIPerfExtension` and plugin registration contracts;
- endpoint, transport, and exporter factory traits;
- plugin package and entry descriptors;
- registry identifiers and category identifiers;
- the host ABI universe identity vocabulary;
- native Rust types that directly cross the library boundary; and
- the shared trait-object representation, method signatures, boundary
  wrappers, ownership rules, and destruction conventions.

It must not depend on transport frameworks, exporter implementations, the CLI,
Tokio, or the orchestration runtime. The host and every conforming plugin link
the SDK-supplied exact compiled artifact selected for their target.

### `aiperf-core`

This crate owns shared product vocabulary and narrow host services needed by
plugins:

- request and response models;
- clock handles and contracts;
- dispatch and observation contracts;
- measurement contracts;
- endpoint body-planning and response-reduction helpers;
- report and metric types; and
- other narrowly reviewed types that both host and plugin must interpret.

It does not own the application bootstrap, CLI, scheduler orchestration,
cellular orchestration, dynamic loader, or aggregate runtime registry.

Because exact ABI-facing compiled artifacts participate in the common host ABI
universe, frequently changing implementation code must not be placed in this
crate merely for convenience. If a concern does not need to cross the boundary,
it belongs in the host runtime, a category SDK, or a plugin-private crate.

### `aiperf-plugin-sdk`

This crate and its associated build tooling own plugin construction and
conformance:

- plugin declaration and entry-symbol macros;
- manifest and JSON Schema generation;
- host-universe and plugin-build-record generation;
- artifact-closure inspection and validation;
- compiler, linker, allocator, panic, and export-policy enforcement;
- the `cargo aiperf-plugin build --release` workflow; and
- the plugin conformance test harness.

The installed SDK bundle also contains the exact compiler/target identity,
toolchain description, allowlisted sources and versions, prebuilt ABI-facing
`.rlib` artifacts, allocator-provider requirement, host ABI universe record,
hermetic build policy, linker policy, and schemas.

The SDK is not itself one runtime shared library. Its Rust library or proc-macro
pieces are compile/link inputs; its command-line tooling runs at plugin build or
inspection time.

### `aiperf-endpoint-sdk`

This crate contains endpoint-category helpers with an isolated dependency
surface. Examples include endpoint configuration validation, request-body
construction helpers, response interpretation, and endpoint companion-binding
support that does not belong in the universal API crate. It does not define
factory traits or concrete boundary values; those live in
`aiperf-plugin-api`/`aiperf-core`.

Only endpoint plugins that use these helpers link them. A helper's concrete
plugin-private types must not cross into the host unless they are deliberately
promoted into the common ABI universe.

### `aiperf-transport-sdk`

This crate contains transport-category helpers and narrow direct-execution
services. Only transport plugins that use those helpers link them. HTTP, gRPC,
WebSocket, dry-run, and Dynosim implementation dependencies remain in their
measured plugin dependency islands rather than entering universal core by
default. Host/plugin service traits and execution-shape values live in
`aiperf-plugin-api`/`aiperf-core`; this SDK implements helpers against them.

### `aiperf-export-sdk`

This crate contains exporter-category helpers, including generic capture/fold
algorithms and artifact-policy helpers. The capture and artifact service
interfaces themselves live in `aiperf-plugin-api`/`aiperf-core`. Only exporter
plugins that use the helpers link them. Backend clients and implementation
libraries for Parquet, OTLP, MLflow, and W&B remain private to their respective
plugin packages.

All three category SDK artifacts are plugin-private in generation 1. No
category-SDK-defined concrete type may appear in a boundary signature,
trait-object vtable, allocation/drop contract, or host-owned stored value. If a
future design promotes one, its single definition must move into API/core and
the resulting API/core artifact change becomes universe-wide; mixing boundary
types and selectively rebuilt private helpers in one category SDK artifact is
not conforming.

### `aiperf-plugin-host`

This is the dedicated host-side plugin lifecycle crate. Tasks 10 through 16
place strict manifest decoding and normalization, no-follow package acquisition,
static native inspection, discovery authority and priority resolution, loader
and residency ownership, transactional registration/freeze, and canonical lock
generation/diffing here. It depends only on the plugin API/core/SDK layer and
must not depend on `aiperf-runtime`.

The runtime consumes its already validated, frozen registry view and retains
execution orchestration. The CLI constructs and wires the plugin host and
runtime during bootstrap. A change confined to `aiperf-plugin-host` therefore
rebuilds the host executable but does not rebuild plugin libraries unless the
same change also modifies an API/core universe input or emitted build contract.

### `aiperf-runtime` and `aiperf-cli`

These are host-side crates, not plugin SDK crates. They retain process-owned
facilities:

- application bootstrap and plugin discovery/loading;
- registry staging, validation, priority resolution, and freezing;
- scheduling, admission, cancellation, drain, and orchestration;
- worker and cellular construction;
- CLI parsing and command routing; and
- process-global state and lifecycle ownership.

Plugins may use narrow native trait handles supplied by the API/core contracts,
but they must never depend directly on `aiperf-runtime` or receive the complete
orchestration objects.

## Runtime linkage

Every plugin exports one native Rust entry symbol,
`aiperf_plugin_entry_v1`. After validating the immutable package, embedded
records, and complete native dependency closure, the host obtains that symbol
from the exact library handle and calls it as an ordinary Rust-ABI function.
Factory and runtime dispatch then use ordinary native Rust trait objects.

There is no C DTO, serialized request, function-table facade, marshalling
layer, or per-call allocation wrapper at the plugin boundary.

All cross-boundary `Global` allocation uses the exact same
`aiperf_alloc_v1` provider. The host and plugins directly import that provider
through their Rust `GlobalAlloc` shims. The provider is a shared native library
because allocation may occur on one side of the boundary and destruction on
the other; it is not an AIPerf function-table wrapper around every allocation.

## The two build identities

The rebuild boundary is expressed through two separate identities.

### Host ABI universe identity

`host_abi_universe_id` identifies facts that must be byte-identical between the
host and every plugin for one target. These include:

- plugin API generation and source API version;
- exact Rust compiler and sysroot artifacts;
- target specification, pointer width, and endianness;
- panic strategy and codegen backend;
- ABI-affecting compiler flags and configuration;
- exact ABI-facing compiled-crate artifacts;
- participating proc-macro artifacts;
- allocator-provider contract, loader identity, and artifact digest;
- boundary linker topology policy; and
- target system-library policy.

Changing one of these facts creates a new universe. The host rejects an old
plugin before native activation when its embedded universe ID differs.

### Plugin artifact build identity

`plugin_artifact_build_id` identifies one plugin's private build. Its record
includes:

- the common `host_abi_universe_id`;
- plugin package name and version;
- plugin source and private feature inputs;
- normalized compiler and linker invocations;
- build scripts, generated sources, and hermetic environment;
- the plugin link payload;
- the final plugin loader identity; and
- the complete distribution-controlled non-system native dependency closure.

A private plugin change produces a new plugin build ID without changing the
host universe or unrelated plugins.

## Recompilation matrix

| Changed input | Host executable | Changed plugin | Same-category consumers | Unrelated plugins | Allocator provider |
|---|---:|---:|---:|---:|---:|
| One plugin's private Rust source | No | Yes | No | No | No |
| One plugin's private Rust dependency | No | Yes | No | No | No |
| One plugin's private native dependency closure | No | Yes | No | No | No |
| Plugin-private category-SDK helper artifact | No | Yes, if it consumes it | Yes, if they consume it | No | No |
| Host orchestration implementation only | Yes | No | No | No | No |
| `aiperf-plugin-api` compiled artifact | Yes | Yes | Yes | Yes | No |
| ABI-facing `aiperf-core` compiled artifact | Yes | Yes | Yes | Yes | No |
| SDK-generated ABI/entry/linker glue | Yes | Yes | Yes | Yes | No |
| Participating SDK proc-macro artifact | Yes | Yes | Yes | Yes | No |
| Allocator contract or provider artifact | Yes | Yes | Yes | Yes | Yes |
| Rust compiler, sysroot, target, panic, or ABI flags | Yes | Yes | Yes | Yes | No, unless its artifact changes |
| Build-tool diagnostics or UX only | No | No | No | No | No |
| Existing-entry priority, discovery-root order, or install-generation only | No | No | No | No | No |

“Same-category consumers” means only plugins that actually link the changed
plugin-private category helper. The generation-1 category SDK artifacts contain
no boundary types and are not in the common ABI-facing closure. A future
promotion cannot merely add such an artifact to that closure while retaining
the selective-build claim: the boundary type must move into API/core, and that
API/core change follows the universe-wide row.

## Concrete change examples

### Change W&B batching logic

Rebuild only the W&B plugin and its package build record. The host, Parquet,
OTLP, MLflow, endpoints, and transports remain unchanged.

### Update a private W&B client dependency

Rebuild only the W&B plugin and any affected private native closure members.
The dependency remains private only if none of its types, traits, allocator
behavior, panic behavior, or native handles crosses the host/plugin boundary.

### Change a helper used by Parquet and W&B

If the helper is plugin-private implementation code in `aiperf-export-sdk`,
rebuild the Parquet and W&B plugins. Do not rebuild OTLP, MLflow, transports,
endpoints, or the host. This is valid only because no export-SDK-defined type
crosses the native boundary.

### Change an endpoint factory method signature

The method belongs to the common plugin API. Rebuild the host and every plugin,
not only endpoint plugins, because all loaded libraries must declare the same
host ABI universe.

### Change a shared request or observation type

If the type belongs to the ABI-facing `aiperf-core` artifact, rebuild the host
and every plugin. Private fields, generated code, layout, drop behavior, and
generic wrappers count when both sides instantiate or interpret them, even if
the source-level public signature appears unchanged.

### Change scheduler or CLI implementation code

Rebuild only the host executable when the change does not alter API/core
boundary artifacts, allocator behavior, build policy, or the universe record.

### Change plugin-SDK diagnostics

Rebuild or redistribute only the build tool. Existing plugin binaries remain
valid if generated bytes, validation policy, schemas, embedded records, and the
host universe remain identical.

### Change plugin-SDK entry glue or linker policy

Rebuild the host and every plugin. These changes alter code or policy at the
native boundary and therefore create a new host ABI universe.

### Change priority or discovery installation state

Changing only an existing entry's signed priority, a discovery-root ordering
input, or the selected immutable installation generation does not recompile
code. Reacquire the package, regenerate the canonical installation lock, and
start a new AIPerf process. Plugin selection is frozen at startup; there is no
live registry mutation or hot reload.

That rule does not cover adding, deleting, or changing a manifest capability
entry. A package's declared entry set must exactly equal the registrations
observed from its loaded library. Therefore an endpoint/transport/exporter entry
change normally requires changing and rebuilding that plugin code, followed by
manifest and lock regeneration. A manifest-only entry-set change is rejected;
it is not a no-recompile override mechanism.

## Telemetry packaging boundary

Telemetry is deliberately split by dependency and release surface:

```text
basic exporters -> one small basic-export plugin
Parquet         -> one independent Parquet plugin
OTLP            -> one independent OTLP plugin
MLflow          -> one independent MLflow plugin
W&B             -> one independent W&B plugin
```

Changing OTLP normally rebuilds only OTLP. Updating Parquet normally rebuilds
only Parquet. CI rejects a telemetry package that links another telemetry
backend's implementation dependency. A change to universal exporter boundary
types in API/core remains universe-wide, while a category helper change rebuilds
only its actual consumers.

## Boundary-promotion rule

A dependency or helper remains private only while the host cannot instantiate,
interpret, lay out, drop, or otherwise rely on its concrete values or code. If
a private type becomes concrete at the boundary, or the host begins to rely on
its representation, validity, vtable, auto-traits, drop behavior, panic
behavior, allocator behavior, or native handles, the SDK must promote its exact
compiled-artifact closure into the common host ABI universe.

That promotion intentionally changes the rebuild result from “one plugin” to
“host and every plugin.” It may not be hidden by claiming that the source API
did not change.

## Operational summary

```text
private plugin change
    -> rebuild one plugin

plugin-private category helper change
    -> rebuild only plugins that link the helper

host orchestration change
    -> rebuild only the host

API, ABI-facing core, allocator, toolchain, panic, or boundary-linker change
    -> rebuild the host and every plugin in that target universe
```

The source API SemVer and compiled identity are intentionally separate. A
source-compatible patch or additive minor release can still produce a new host
ABI universe and require all installed plugin binaries to be rebuilt.
