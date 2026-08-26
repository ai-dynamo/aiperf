<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Rust runtime plugins

## Purpose

AIPerf will move its endpoint, transport, and exporter implementations into
native Rust shared-library plugins. A process discovers installed manifests at
startup, eagerly loads every required package and every package that owns a
winning entry, registers their native Rust trait implementations, and freezes
one implementation universe before validation or execution. Fully shadowed or
quarantined optional libraries are cataloged but never executed. The same
mechanism serves first-party and third-party plugins.

The design is intended to isolate compile and link dependency graphs while
preserving the runtime shape of the existing `AIPerfExtension` and
`AIPerfRegistry` seams. It is deliberately closer to Cargo features selected at
process startup than to a language-neutral plugin API.

## Goals

- Load multiple `.so`, `.dylib`, or `.dll` libraries into one AIPerf process.
- Exchange native Rust trait objects and AIPerf types without serialization or
  ABI wrapper calls.
- Apply all plugin registration before runtime, worker, or external-effect
  construction, then keep the registry immutable for the process lifetime.
- Move first-party endpoint, transport, and exporter implementations through the
  same path available to third parties.
- Auto-discover package manifests in the spirit of Python AIPerf's
  `plugins.yaml` and package entry points.
- Permit deterministic priority-based replacement of registered
  implementations.
- Produce measurable clean-build, incremental-build, and link-time improvements.
- Add no abstraction or dispatch layer to request- or token-processing paths and
  introduce no statistically significant benchmark regression.

## Non-goals

- Stable binary compatibility across Rust compilers or arbitrary AIPerf builds.
- A C ABI, language-neutral SDK, WebAssembly runtime, or process-isolated plugin
  protocol.
- Loading, unloading, upgrading, enabling, or disabling plugins after the
  application registry is frozen.
- Sandboxing plugin code. Loaded plugins are trusted native code with the same
  process authority as AIPerf.
- Transferring executable libraries to remote cells in the initial release.
- Making every AIPerf internal module a supported plugin API.

## Normative language and invariants

`MUST`, `MUST NOT`, `REQUIRED`, `SHALL`, `SHALL NOT`, `SHOULD`, `SHOULD NOT`,
and `MAY` are normative. A component that violates a `MUST` or `MUST NOT` is
not a conforming implementation of this design.

The following invariants apply together. None is optional:

1. **Native Rust boundary.** The loader and plugins exchange the exact Rust
   traits and Rust values declared by the AIPerf plugin API. There is no C ABI,
   `abi_stable` facade, serialization layer, generated function table, Python
   runtime, or process RPC between a factory and its host.
2. **Exact binary identity.** A library is callable only after its manifest's
   ABI fingerprint exactly equals the host fingerprint and its declared BLAKE3
   digest exactly equals the acquired library bytes. Source compatibility does
   not waive binary mismatch.
3. **Load before effects.** Plugin discovery, compatibility filtering, library
   loading, registration, priority resolution, and registry freezing complete
   before clocks, Tokio runtimes, workers, network clients, artifact
   directories, cells, or benchmark side effects are created.
4. **One frozen universe.** Capabilities, validation, execution, re-executed
   children, controllers, and cells use one resolved plugin-lock identity. No
   execution path constructs a separate built-in registry or discovers an
   additional plugin.
5. **Process-lifetime residency.** A library whose code has been called is never
   unloaded or replaced before process exit. No trait object's vtable or future
   may outlive its defining code.
6. **No runtime mutation.** Registration is impossible after freeze. There is no
   reload, unload, enable, disable, override, or priority change during a run.
7. **Transactional packages.** A package contributes all declared entries or
   none. A registration error cannot expose a prefix of the package.
8. **Deterministic override.** Priority is resolved per normalized
   `(category, name)`. Filesystem order, discovery-source order, and load order
   never break a tie.
9. **First-/third-party parity.** First-party endpoints, transports, and
   exporters use the same manifest, ABI, entry-symbol, registration, priority,
   and freeze mechanisms as third-party implementations. The host has no hidden
   static preference path.
10. **Core reuse.** Plugins can use published AIPerf API, core, and SDK crates.
    They cannot depend on the orchestration runtime or instantiate host-owned
    global services.
11. **Hot-path shape.** Dynamic loading adds no wrapper call, serialization,
    buffer conversion, allocation, lock, thread hop, or IPC operation to the
    existing request/token paths. Existing native trait-object calls remain the
    only category dispatch.
12. **Measured equivalence.** A first-party implementation does not leave its
    static fallback until the performance gates in this document pass.
13. **Open selection.** Config v2 selects transports and exporters through
    normalized registry IDs plus plugin-owned strict configuration. A closed
    Rust enum cannot prevent selection of a successfully registered plugin.
14. **Trusted code.** Auto-discovery is not a sandbox. Loading a plugin grants
    it AIPerf's process authority. Inspection commands distinguish reading a
    manifest from executing library code.
15. **No silent runtime fallback.** Once a catalog is frozen, failure of its
    selected implementation fails the operation. A shadowed implementation is
    never promoted during validation or execution.

### Decision traceability

This table is the authoritative record of the design decisions made during
review. Later edits MUST update both the detailed section and this table when a
decision changes.

| Decision | Normative resolution | Detailed section |
|---|---|---|
| Artifact form | Multiple native Rust `.so`/`.dylib`/`.dll` libraries loaded into one process | Library contract |
| Programming model | Native Rust traits and types; no C ABI or ABI-wrapper facade | Invariant 1; rejected alternatives |
| Compatibility meaning | Source API is SemVer; binary compatibility requires an exact ABI fingerprint and rebuild on change | Compatibility contract |
| Runtime lifecycle | Eagerly load active packages, register transactionally, freeze once, retain code until process exit | Composition and lifecycle |
| Initial categories | Endpoint, transport, and exporter only | Runtime category behavior |
| First-party behavior | First-party implementations migrate to the same plugin path as third parties | Invariant 9; migration |
| Packaging granularity | A library may register one or many entries; grouping follows dependency size/coupling | Library contract |
| Telemetry packaging | Parquet, OTLP, MLflow, and W&B are separate dependency artifacts, not one telemetry bundle | Library contract; migration |
| Discovery experience | Strict `plugins.yaml` manifests are auto-discovered from defined installation paths | Manifest format; discovery and priority |
| Override behavior | Per-entry signed priority; unique maximum wins; equal maximum is ambiguous | Discovery and priority |
| Core reuse | Plugins depend on allowlisted API/core/category SDK crates, never the orchestration runtime | Crate architecture |
| Performance | No added hot-path operations plus normative statistical equivalence gates | Performance contract |
| Multi-process consistency | Parent, re-exec children, controllers, and cells require the exact same plugin-lock digest | Composition and lifecycle |
| Security | Native plugins are trusted code; digests identify but do not sandbox them | Failure and trust policy |

## Considered and rejected alternatives

### Stable C ABI and generated function tables

A versioned C ABI could keep a plugin binary compatible across multiple AIPerf
or Rust releases. It was rejected because it would replace native AIPerf traits
with FFI-safe DTOs and function tables, require explicit ownership and async
bridges, and add a second dispatch/marshalling surface. Plugin authors could be
given a Rust wrapper, but the runtime boundary would still not be native Rust.
That conflicts with the primary design requirement.

### `abi_stable` or another Rust-authored stable-ABI facade

This improves author ergonomics over handwritten C declarations but retains the
same ABI-safe wrapper model, restricted container types, and conversion rules.
It was rejected for the same native-trait and hot-path reasons. It remains a
possible design for a different product that prioritizes cross-release binary
compatibility over this design's requirements.

### Process plugins over protocol v2, RPC, or IPC

A process boundary would provide crash isolation and a stable serialized
contract. It was rejected because independently composing an endpoint,
transport, and exporter would introduce serialization, scheduling, or IPC into
the request lifecycle. Delegating a complete benchmark to one process avoids
hot-path IPC but creates indivisible engine distributions rather than the
requested in-process capability composition.

### One aggregate distribution library

A generated library containing the complete selected feature set would reduce
the number of loaded artifacts and permit more optimization inside the
aggregate. It was rejected as the primary architecture because changing one
third-party or optional dependency would rebuild the aggregate, packages could
not be independently installed, and priority composition would happen at build
time rather than startup. The plugin contract still permits one author to bundle
several related entries in one library.

### Static Cargo features or multiple complete executables

Static features provide the strongest whole-program optimization. Multiple
prebuilt executables can also isolate feature sets without FFI. Both were
rejected because adding or changing a plugin requires relinking or selecting a
different complete AIPerf distribution; neither permits multiple independently
installed packages to compose into one frozen process registry.

### Rust `cdylib`

`cdylib` is intended to expose a foreign-language ABI and suppresses Rust crate
linkage metadata. It was rejected in favor of Rust `dylib`, because the selected
contract intentionally exchanges native Rust traits and types and accepts exact
toolchain coupling.

### Loading arbitrary libraries found in a directory

Scanning for filename extensions and calling any matching library was rejected.
It provides no side-effect-free catalog, digest, priority, compatibility, or
declared-capability information and makes a stray file executable. Discovery
therefore scans strict manifests only.

### Explicit-path-only discovery

Requiring every library path on every invocation is reproducible but does not
match the install-and-discover experience of Python AIPerf's plugin entry
points. It was rejected as the sole mechanism. Explicit paths remain an
additive override and diagnostic facility.

### Duplicate rejection without priority

The current Rust registry's unconditional duplicate rejection is simple and
safe, but it prevents an installed package from deliberately replacing a
first-party implementation. It was rejected in favor of explicit per-entry
priority. Equal-priority ambiguity still fails closed.

### Dynamic reload or unload

Reload would require proving that no registry clone, factory product, future,
callback, or vtable remains reachable. The current application intentionally
shares registry and factory handles broadly. Reload was rejected because it
would complicate ownership and introduce synchronization without serving the
compile-time or third-party installation goals.

### Plugins depending on the complete orchestration runtime

Allowing a plugin to depend directly on today's `aiperf-runtime` monocrate would
recreate the large compile graph, expose private implementation details, risk
duplicate global state, and create a dependency cycle between loader and
loaded code. It was rejected in favor of focused downward API/core/SDK crates.

## Compatibility contract

The public plugin API is source-stable within its declared compatibility policy,
but the binary ABI is exact-build only. Rust does not stabilize the ABI of trait
objects, standard-library containers, async futures, enums, or unwinding across
independent compiler releases. A plugin therefore loads only when its complete
Rust ABI fingerprint equals the host fingerprint.

The fingerprint covers:

- Rust compiler commit and target triple;
- pointer width, endianness, panic strategy, and ABI-relevant code-generation
  settings;
- the plugin API generation;
- the identities and enabled features of `aiperf-plugin-api` and every
  ABI-facing AIPerf core crate;
- ABI-relevant dependency versions.

The SDK computes the fingerprint rather than asking authors to maintain it. An
AIPerf upgrade that changes the fingerprint requires rebuilding the plugin. A
human-readable API version remains separate from the fingerprint so diagnostics
can distinguish a deliberate API revision from a compiler or build mismatch.

"Stable" in this design means that the published Rust source API follows
AIPerf's compatibility policy and that incompatible binaries are diagnosed
before execution. It does **not** mean that one compiled library survives an ABI
fingerprint change. Documentation and diagnostics MUST use the terms "source API
version" and "exact ABI fingerprint" separately and MUST NOT advertise a stable
Rust binary ABI.

The plugin source API has its own SemVer version beginning at `1.0.0`, independent
of the AIPerf product version. A major change may remove or alter public source
contracts. A minor change is additive and retains source compatibility for
conforming plugin crates. A patch change corrects behavior without changing the
documented source signature. This SemVer promise affects recompilation only:
even an additive or patch release may produce a different exact ABI fingerprint,
in which case installed binaries MUST be rebuilt.

The canonical fingerprint input is a length-delimited, sorted record generated
by `aiperf-plugin-sdk`; it is not Cargo's display version alone. Each record
contains the field name and raw value, and the final ID is BLAKE3 over the
canonical bytes. At minimum the record contains:

```text
plugin_api_generation
rustc_commit_hash
rustc_host
target_triple
target_pointer_width
target_endian
panic_strategy
codegen_backend
abi_facing_crate_name -> exact package ID, source digest, enabled feature set
abi_facing_native_dependency -> exact identity when its types cross the boundary
```

The build helper emits the canonical input beside the digest so
`aiperf plugins inspect-abi` can explain the first differing field. Unknown
fingerprint fields are rejected by an older host rather than ignored. Profile
settings that only change optimization level are recorded for diagnostics but
do not create compatibility by themselves; the host distribution policy still
requires first-party plugins to use its named optimized profile.

## Crate architecture

The current runtime monocrate will be split only at measured plugin-facing
boundaries:

```text
aiperf-plugin-api
|-- AIPerfExtension and registration contracts
|-- endpoint, transport, and exporter factory traits
|-- descriptors, registry identifiers, and ABI identity
`-- native types that cross the library boundary

aiperf-core
|-- request and response models
|-- clocks, dispatch, observation, and measurement contracts
|-- endpoint body and response-reduction helpers
`-- report and metric types

aiperf-plugin-sdk
|-- plugin declaration and entry-symbol macros
|-- manifest generation and validation support
|-- ABI inspection helpers
`-- plugin conformance test harness

aiperf-endpoint-sdk
aiperf-transport-sdk
aiperf-export-sdk
`-- category-specific helpers with isolated dependency surfaces
```

The host runtime depends downward on the API and core crates. Plugin crates
depend on the API plus only the core or category SDK crates they need. Plugins
do not depend on the complete orchestration runtime. This avoids dependency
cycles, duplicate process-global ownership, and needless ABI-fingerprint
expansion.

`aiperf-plugin-api` MUST remain the smallest crate in the graph and MUST NOT
depend on a transport framework, exporter backend, CLI parser, Tokio runtime,
or the orchestration crate. ABI-facing model types that are currently defined
inside large modules move either into this crate or a narrowly scoped core
crate. A crate is ABI-facing if a plugin trait method accepts or returns one of
its types, including behind `Box`, `Rc`, `Arc`, `Result`, a future, or another
container.

The workspace MUST publish an explicit API allowlist. A third-party plugin is
supported when it imports only allowlisted AIPerf crates and public items.
Runtime-private crates remain unpublished or expose no plugin-facing public
surface. Adding a crate to the allowlist is an API decision and requires an ABI
fingerprint input; merely making a Rust item `pub` does not make it supported.

Plugins MAY use arbitrary ordinary Rust dependencies internally. Such a
dependency enters the ABI fingerprint only when one of its types, traits,
panics, allocator behaviors, or native handles crosses the host/plugin boundary.
Pure implementation dependencies remain private to that plugin and can change
without rebuilding the host.

The ABI-facing crate set is generated, reviewed, and checked into the SDK
release. It is the transitive closure of allowlisted crates that define any
public item named by a plugin trait signature. The SDK obtains exact package
IDs, sources, and features from Cargo metadata and hashes the defining source
trees. Build scripts cannot remove an ABI-facing crate from the fingerprint.
When review discovers an untracked public type, the API release is invalid until
the allowlist and fingerprint generator include it.

Process-owned facilities remain host-owned and are passed into plugins as
native trait handles. These include clocks, scheduling, observation, artifact
policy, and worker construction context. A plugin may use those services but
must not create a competing process-global instance.

### Rust linkage and allocator rules

The SDK controls the complete plugin build, not just a header crate. A
conforming artifact MUST:

- use Cargo `dylib`, never `cdylib`;
- use the exact compiler and target declared by the host SDK bundle;
- resolve every ABI-facing AIPerf crate to the exact package identity used by
  the host;
- use the host distribution's panic strategy;
- not declare a `#[global_allocator]`;
- not export a second copy of an AIPerf process-global singleton;
- include a linker/dependency map that the manifest generator validates.

The host executable remains the authority for the process allocator (mimalloc
in the shipped AIPerf binary). Plugin-created trait objects are destroyed through
their plugin-defined drop vtables while the library is resident. Native owned
values crossing by value use the same process allocator contract. Windows,
Linux, and macOS conformance tests MUST prove cross-boundary allocate/return/drop
for every owned ABI-facing container family before that family can enter the
public plugin API.

The API SHOULD avoid `Any` and host-side downcasts across the library boundary.
Validated plugin configuration remains an opaque native trait object and is
returned to the same defining factory for consumption. If an `Any` downcast is
unavoidable, its concrete type's defining crate, build identity, and
cross-library `TypeId` behavior become fingerprinted and require a dedicated
cross-platform conformance test.

### Supported author workflow

The supported third-party workflow is one command owned by the SDK, exposed as
`cargo aiperf-plugin build --release`. It obtains the host compatibility record
from an installed AIPerf SDK bundle or an explicit `--sdk` directory, verifies
the exact Rust toolchain, invokes Cargo with the required crate type and linker
policy, validates dependencies and allocator symbols, emits the manifest, then
hashes the final library into that manifest. Handwritten manifests and ordinary
`cargo build` outputs may be inspected, but they are not conforming distributable
artifacts until the SDK validator accepts them.

The SDK bundle contains the allowlisted crate versions/sources, target identity,
toolchain file, plugin API compatibility record, platform linker policy, and
JSON Schema. It does not include or expose orchestration-private crates. A
plugin can use `aiperf-core` and category SDK helpers as normal Rust dependencies
and receives ordinary compiler type checking and documentation for them.

## Library contract

Each plugin is a native Rust `dylib` and exports exactly one required unmangled
Rust entry symbol, `aiperf_plugin_entry_v1`. It is a Rust-ABI function, not an
`extern "C"` function:

```rust
pub type PluginEntryV1 = unsafe fn() -> PluginDeclarationV1;

pub struct PluginDeclarationV1 {
    pub package: PluginPackageDescriptor,
    pub extension: Box<dyn AIPerfExtension>,
}

pub trait AIPerfExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>)
        -> Result<(), ExtensionError>;
}
```

The exact fields of the two public structs belong to
`aiperf-plugin-api`; the sketch above fixes the ownership and call shape. The
SDK's declaration macro emits the symbol, embeds the package descriptor, and
prevents authors from selecting a different symbol name. The loader calls the
symbol only after sidecar manifest digest and fingerprint validation. The
loader's unsafe code is confined to acquired-library residency, symbol
resolution, and this initial Rust-ABI call.

`PluginPackageDescriptor` repeats package name, version, source API version, and
ABI fingerprint. The loader MUST compare every repeated field with the manifest
after the entry call. A mismatch rejects the entire package. This second check
detects accidental manifest/library pairing errors; it is not a sandbox against
malicious native code.

`PluginRegistrar` is a manifest-bound facade over a private staged registry. It
supplies package identity and priority from the manifest, observes every actual
registration, and exposes only the endpoint, transport, and exporter categories
in this version. A plugin cannot directly mutate the aggregate registry or
claim a different origin.

The extension registers ordinary `EndpointFactory`, `TransportFactory`, and
`Exporter` trait implementations. One library may register any number and mix
categories freely. Packaging granularity is consequently a build and dependency
decision, not an API constraint.

First-party packages should normally follow cohesive dependency islands rather
than one library per implementation. Small basic exporters may share one
library, while Parquet, OTLP, MLflow, and W&B remain separate because their
dependency and release surfaces differ. HTTP, gRPC, WebSocket, and Dynamo may
likewise be packaged according to measured build size and coupling.

Every object created from plugin code is destroyed while its library remains
resident. The process-global `LoadedLibrarySet` intentionally retains library
handles until operating-system process teardown, including after an
`Application` value is dropped. `Application` owns the frozen registry and
ordinary plugin-created runtime objects; dropping them cannot unload code. This
is stronger than relying on Rust struct-field drop order.

## Manifest format

Each package installs one `plugins.yaml` adjacent to its library:

```yaml
schema_version: "2.0"

plugin:
  name: aiperf-export-otlp
  version: "0.12.0"
  api_version: "1.0"
  library: aiperf_export_otlp
  digest: "blake3:<library-digest>"
  abi_fingerprint: "blake3:<abi-fingerprint>"

exporter:
  otlp:
    priority: 0
    description: OpenTelemetry report exporter
    metadata: {}
```

Multiple packages in one discovery directory use package-qualified manifest
names while retaining the same schema, for example:

```text
plugins.d/
|-- aiperf-http.plugins.yaml
|-- libaiperf_http.so
|-- aiperf-export-otlp.plugins.yaml
`-- libaiperf_export_otlp.so
```

The macOS and Windows layouts substitute their platform library filenames. A
package-specific directory explicitly added with `--plugin-path` may use the
unqualified `plugins.yaml` name.

The manifest is declarative inventory. It does not name Rust types or select
individual symbols. The fixed package entry symbol performs registration. The
loader derives the platform filename from `library`; installed manifests cannot
use an absolute path or escape their package directory.

Manifest schema `2.0` is strict: unknown root, package, category, entry, or
metadata fields are rejected unless the relevant category schema explicitly
declares them. Required package fields are `name`, `version`, `api_version`,
`library`, `digest`, and `abi_fingerprint`. At least one supported category
entry is required. Package and entry names use the existing `RegistryId`
normalization rules; the manifest retains the authored spelling for display but
all identity and conflict operations use the normalized value.

`library` MUST be one basename without a directory separator, extension, drive
prefix, or platform prefix. The loader maps `example` to `libexample.so` on
Linux, `libexample.dylib` on macOS, and `example.dll` on Windows. The resulting
file MUST be a regular file in the manifest's package directory. Symlink and
canonicalization policy is uniform across discovery sources: the loader resolves
the package directory once, rejects a resolved library outside it, records file
identity and metadata, hashes the file immediately before load, and rejects a
metadata identity change observed immediately after load. The digest is a
reproducibility check, not protection from an attacker who can modify the plugin
installation concurrently with execution.

Every category entry contains `priority` as a signed 32-bit integer with default
zero, `description` with default empty text, and category-specific `metadata`
with a strict schema. Priority has no package-level default other than zero and
cannot be changed by the entry function. Aliases, when a category supports
them, are declared as a sorted unique list and participate in conflict
resolution exactly like canonical IDs; an alias cannot replace a canonical ID
implicitly.

Category metadata supports discovery, help, generated documentation, and
side-effect-free inspection without loading code. After loading, the host
requires the extension's actual category/name registrations to match its
manifest exactly. An undeclared registration, missing registration, category
mismatch, or priority mismatch rejects the complete package.

## Discovery and priority

The loader scans YAML manifests, never arbitrary libraries. The concrete
discovery sources are:

1. the distribution directory recorded at build/install time, ending in
   `aiperf/plugins.d`;
2. on Linux, each `$XDG_DATA_DIRS/aiperf/plugins.d` entry (defaulting to
   `/usr/local/share/aiperf/plugins.d:/usr/share/aiperf/plugins.d`) followed by
   `$XDG_DATA_HOME/aiperf/plugins.d` (defaulting to
   `$HOME/.local/share/aiperf/plugins.d`);
3. on macOS, `/Library/Application Support/AIPerf/plugins.d` followed by
   `$HOME/Library/Application Support/AIPerf/plugins.d`;
4. on Windows, `%PROGRAMDATA%\AIPerf\plugins.d` followed by
   `%LOCALAPPDATA%\AIPerf\plugins.d`;
5. directories in `AIPERF_PLUGIN_PATH`, parsed with the platform path-list
   separator;
6. each repeatable `--plugin-path <directory>` and
   `--plugin-manifest <file>` argument in authored CLI order.

Missing default directories are ignored. An unreadable existing directory,
invalid `AIPERF_PLUGIN_PATH`, or invalid explicit path produces a diagnostic;
explicit CLI paths are fatal. `--no-auto-plugins` disables sources 1 through 4
but does not disable environment or explicit sources. Internal re-execution
does not rediscover from mutable CLI or environment inputs; it consumes the
parent's plugin lock and uses discovery only to locate the exact locked package
identities.

Directory entries and normalized manifest identities are sorted before
resolution, so filesystem enumeration order has no semantic effect. Repeated
identical package manifests are deduplicated. Different manifests claiming the
same package identity are rejected.

Only files named `plugins.yaml` or ending in `.plugins.yaml` are candidates.
Discovery is non-recursive. Manifest path order is retained solely for
diagnostics; it never decides a winner. Package identity is the normalized
package name plus exact package version. Two manifests with that identity are
identical only when their canonical decoded forms and library digests match.

Priority is resolved independently for each normalized `(category, name)`:

- the highest compatible priority wins;
- an equal-priority tie is ambiguous and makes that component unavailable;
- external plugins must declare a higher priority to replace a first-party
  implementation;
- the frozen catalog records the winner and every shadowed or unavailable
  candidate.

The resolution algorithm is fixed:

1. Strictly decode all candidate manifests.
2. Remove candidates whose manifest, platform artifact, digest, or ABI
   fingerprint is invalid; record them as quarantined and do not load them.
3. Group remaining declared entries by normalized `(category, name)`.
4. Find the maximum signed priority in each group.
5. If exactly one entry has that priority, mark it the manifest winner and every
   lower entry shadowed. If more than one has that priority, mark the key
   ambiguous and select no winner.
6. Load every required package and every package that owns at least one manifest
   winner. A fully shadowed optional package is not executed.
7. Validate each loaded package's actual registrations. If a package fails,
   mark the process composition poisoned, retain every already loaded library,
   and fail the composition. Do not promote a lower-priority candidate or retry
   in that process.
8. Freeze successful winners, ambiguities, shadows, and quarantine causes into
   one catalog. Selection of an ambiguous or unavailable key fails with all
   relevant candidates and causes.

Step 7's no-promotion rule prevents a broken override from silently changing a
run to a different implementation. A later fresh process may obtain a different
catalog only after the installation or authored plugin inputs change.

Registration is manifest-bound. When an extension registers a factory, the
registry obtains its priority and package provenance from the active package
context rather than trusting plugin-supplied ad hoc values. The package is
staged transactionally; an error commits none of its entries. Priority
resolution then produces one immutable winner per category and name.

Required first-party packages are declared by the signed or packaged AIPerf
distribution inventory, not by a plugin's own manifest. A third-party manifest
cannot mark itself required. An explicit `--plugin-manifest` makes that package
required for the invocation. The distribution inventory contains exact package
names, versions, library digests, ABI fingerprints, and component keys but does
not pin a local absolute path. Installation updates the inventory and its
manifest/library artifacts atomically; a partially installed required package
is a broken distribution and fails composition.

## Composition and lifecycle

Application composition follows a strict sequence:

1. Discover and strictly decode manifests.
2. Classify incompatible and malformed optional packages without executing
   them.
3. Resolve compatible candidates and priority outcomes.
4. Acquire each required library, hash its bytes, and compare its manifest ABI
   fingerprint with the host.
5. Load the library with restricted platform dependency-search rules.
6. Resolve and invoke `aiperf_plugin_entry_v1`.
7. Apply manifest-bound transactional registration.
8. Verify declared and actual registrations.
9. Freeze `AIPerfRegistry` and derive the capabilities catalog from it.
10. Construct runtimes, workers, and any effectful resources.

These states are represented by distinct Rust types:

```text
DiscoveredCatalog -> CompatibleCatalog -> LoadedCatalog
                  -> RegistryBuilder -> FrozenAIPerfRegistry -> Application
```

Mutation methods exist only on `RegistryBuilder` and the package-scoped
`PluginRegistrar`. `FrozenAIPerfRegistry` exposes lookup and catalog methods but
no registration or mutable category accessors. Freezing consumes the builder;
there is no thaw operation.

Production composition is process-global. The first successful load installs a
process-resident `LoadedLibrarySet` with its plugin-lock digest. A second
composition request in the same process may reuse it only when the requested
lock digest is identical; a different digest is an error. Library handles are
intentionally retained until operating-system process teardown and are never
passed to `Library::close` or dropped during normal Rust destruction. Tests that
exercise loading, failure, or differing catalogs run in subprocesses. This rule
eliminates dependence on registry-clone or factory-product drop ordering.

If failure occurs after any entry symbol is called, the process-global loader is
marked poisoned. All acquired handles remain resident, every later composition
attempt returns the original failure, and the process cannot execute a
benchmark. This prevents retrying against native code that may have changed
process-global state before it failed.

Platform dependency loading is constrained as follows:

- Linux plugins resolve private native dependencies through an authored
  `$ORIGIN` runpath in the plugin package; the loader does not mutate global
  `LD_LIBRARY_PATH`.
- macOS plugins use `@loader_path`-relative install names; the loader does not
  mutate `DYLD_LIBRARY_PATH`.
- Windows uses restricted `LoadLibraryEx` search flags and a package-scoped DLL
  directory rather than the current working directory or ambient `PATH`.

The SDK validates these policies in the produced artifact. A dependency that
cannot be resolved under the restricted policy rejects the package. A plugin
MUST NOT rely on the process current directory for code or native dependencies.

No execution path may construct a fresh built-in registry. Root help and
manifest-only inspection need not load libraries. Capability validation and
execution use the exact frozen `PluginUniverse`.

The command behavior is explicit:

- root `--help` and shell-completion generation do not discover or load plugins;
- `aiperf plugins list` discovers and decodes manifests without executing code;
- `aiperf plugins validate` performs digest, ABI, dependency, entry-symbol, and
  registration checks and therefore executes trusted plugin initialization;
- `aiperf config`, profile validation, execution, and native evaluation compose
  the frozen catalog before resolving any registered component;
- commands unrelated to registered runtime components do not load plugins
  unless they request the capability catalog.

The parent records resolved package identities, priorities, artifact digests,
and ABI fingerprints in an internal plugin lock. Re-executed children must
reproduce that lock exactly. Cellular controllers bind its digest into cell
registration and refuse a cell with a different plugin universe before the run
barrier. Remote hosts must preinstall the required packages in the initial
release.

The canonical plugin lock contains schema version, ordered package identities,
library digests, ABI fingerprints, every category winner and priority, every
ambiguity, and required-package identities. It excludes absolute paths. The lock
digest is BLAKE3 over canonical length-delimited bytes, not YAML or JSON map
serialization order. Parents transfer the exact lock to same-host children over
the existing private bootstrap mechanism. Cross-host launch metadata carries
the lock digest; each process rebuilds the canonical lock from its local
installation and proves equality before registration completes. A mismatch
names the first differing package or component and prevents barrier advance.

Automatic executable transfer is intentionally absent. Kubernetes, SLURM, and
other cross-host distributions MUST package the same plugin artifacts on every
participating host or image. The controller never accepts a merely compatible
but byte-different plugin lock.

### Panic and ownership behavior

Entry and registration calls run inside `catch_unwind` only when the exact
fingerprint declares `panic=unwind`. A caught panic rejects the package and the
process performs no benchmark action. `panic=abort`, an explicit abort, loader
termination, undefined behavior, or memory corruption cannot be recovered and
may terminate the process; the system makes no isolation claim.

No borrowed value received from a plugin may be retained beyond the lifetime
declared by its native trait method. Any `'static` descriptor from a plugin is
valid because the defining library is process-resident. Futures and callbacks
created by a plugin retain native ownership and finish or are dropped before
process teardown. A plugin MUST NOT spawn a detached task or thread whose code
can run after AIPerf has declared application shutdown complete.

## Runtime category behavior

Only endpoint, transport, and exporter are dynamically loadable in API
generation 1. Dataset loaders, samplers, workloads, actuators, native-graph
factories, clocks, observers, and direct execution seams remain host-owned or
statically composed. Adding a fourth dynamic category requires a new reviewed
API generation or an explicitly backward-compatible source API addition plus
new manifest schema, conformance, lifecycle, and performance coverage.

The manifest describes discoverable capabilities, but executable descriptors
returned by native factories remain authoritative after successful
registration. Composition rejects a discrepancy rather than merging metadata
from two sources. Configuration never selects a Rust type name, symbol name,
package path, or library path; it selects a normalized category ID from the
frozen catalog.

### Endpoints

Endpoint plugins register the native `EndpointFactory` contract. Each worker
prepares its existing worker-local `PreparedEndpoint` once. Formatting and
streaming response parsing continue to use native AIPerf values and byte
buffers. No FFI container, serialization, or additional adapter call is added.

Endpoint canonical IDs and aliases remain open strings. Duplicate and priority
resolution includes aliases before freeze, so a selected alias identifies one
unambiguous winning factory. Generic endpoint policy remains host-owned;
endpoint-specific configuration validation and request/response formatting
remain plugin-owned. Streaming parsers MUST continue preserving incomplete byte
sequences across chunks and MUST emit observations through the existing host
observer seam.

### Transports

Transport plugins register `TransportFactory` and native execution bindings.
The host retains scheduling, admission, phase orchestration, clock ownership,
observation, reduction, and measurement. The plugin owns transport-specific
connection pools, dispatchers, and request executors using the existing
worker-local `Rc` and `LocalSet` discipline.

The currently closed built-in transport configuration becomes an open
registry-selected record:

```yaml
transport:
  id: http
  config:
    protocol: auto
```

Known CLI flags continue to project to registered IDs and plugin-owned raw
configuration. Unknown IDs and plugin-owned unknown fields fail before
preparation.

The canonical authored representation is exactly `{ id, config }`, where `id`
is required and `config` defaults to an empty object. The plugin factory receives
the raw JSON object once during strict startup validation and returns an opaque
native validated-config trait object. The host passes that object only to the
same winning transport factory. No host switch on a closed transport enum may
remain in validation, preparation, execution binding, capability reporting, or
cellular projection.

The transport plugin does not own phase scheduling, request admission, worker
partitioning, metrics, cancellation policy, or clocks. It owns only the
transport-specific execution objects already represented by the transport
traits. A transport that needs a new host capability must add an explicit
capability contract; it cannot reach into runtime-private state.

### Exporters

Exporter plugins receive the finalized native report and host-owned artifact
policy after execution. Fixed exporter enablement fields become an ordered open
selection:

```yaml
exporters:
  - id: json
  - id: otlp
    config:
      endpoint: http://collector:4318
```

Existing CLI and Config-v2 inputs retain their external behavior through
projection into this form. Each selected factory strictly decodes its own
configuration.

The canonical exporter representation is an authored ordered list of
`{ id, config }`, with an empty object default for `config`. Duplicate exporter
IDs in one run are rejected unless that exporter's public descriptor explicitly
declares multi-instance support and defines an instance key. Registry priority
selects an implementation; list order does not override package priority.

Exporter execution order is still host-owned. Each exporter descriptor declares
its existing order band and stable tie-break key. The host sorts enabled
instances deterministically and preserves the rule that local artifact writers
complete before uploaders. A plugin cannot access paths outside the host-created
artifact directory through an unchecked join API.

### CLI compatibility

Existing public flags for first-party components remain accepted for the
migration's supported compatibility window. Projection produces the canonical
open records before protocol-v2 validation. A flag that configures a plugin not
present in the frozen catalog fails with the missing component ID and the
relevant package/quarantine diagnostics. New third-party configuration is
authored through Config v2 until a general `--plugin-config` CLI surface is
separately designed; this specification does not create an untyped flag syntax.

## Performance contract

The dynamic design must satisfy both a structural and measured performance
contract.

Structurally:

- no ABI wrapper, serialization, or buffer conversion exists between host and
  plugin;
- no additional call layer is added to request or token paths;
- the registry continues to return the same native trait-object shapes used by
  current built-ins;
- factories execute only during application or worker preparation;
- worker-local state remains local and gains no synchronization requirement;
- timing and measurement continue through the host-provided `Clock`;
- plugin libraries use the same optimized build profile as the distribution.

Separate shared libraries prevent fat LTO from optimizing across their
boundaries, so literal machine-code identity is not promised. Instead, a
preserved monolithic baseline and the plugin distribution run identical
mock-server workloads. Predefined statistical equivalence bands cover
throughput, TTFT, inter-token latency, CPU time, and allocations. Endpoint
formatting and streaming reduction microbenchmarks prevent network variance
from hiding local regressions. A statistically significant regression outside
the bands blocks migration of that component.

The initial acceptance protocol is normative:

- compare optimized static-baseline and dynamic-plugin artifacts built from the
  same source revision, compiler, target, profile, affinity, and dependency lock;
- run on an otherwise idle pinned host with fixed CPU frequency policy and the
  in-repo mock server;
- execute at least 30 paired samples after warmup for each representative HTTP
  non-streaming, HTTP streaming, gRPC, multi-worker, and exporter workload that
  applies to the migrated component;
- use paired bootstrap 95% confidence intervals over the dynamic/static ratio;
- define throughput ratio as `dynamic / static` and require the lower endpoint
  of its 95% interval to be at least `0.99`;
- define each latency and CPU ratio as `static / dynamic` and require the lower
  endpoint of its 95% interval to be at least `0.99` for TTFT p50/p90/p99,
  inter-token-latency p50/p90/p99, and CPU time per successful request
  (equivalently, dynamic may not be more than 1% worse);
- require no increase in allocations per request in deterministic endpoint and
  reduction microbenchmarks;
- require no new lock acquisition, allocation, serialization, copy, task spawn,
  channel operation, or dynamic-dispatch layer in a code-path inspection of the
  migrated request/token path.

A benchmark with a coefficient of variation above 2% for the primary metric is
invalid, not a pass; it is rerun after diagnosing environmental noise. The
harness records raw samples, environment identity, and confidence intervals as
CI artifacts. Threshold changes require an explicit design/specification
change; a migration patch cannot loosen them to obtain a green result.

Before the first dynamic migration, the repository adds
`rust/benchmarks/plugin-parity.yaml` as the immutable comparison inventory. It
contains at least HTTP non-streaming at concurrency 1 and 64, HTTP streaming at
concurrency 1 and 64 with 32 deterministic response chunks, gRPC unary and
streaming at concurrency 1 and 64, a four-worker run, and an exporter pass over
100,000 deterministic records. Each case performs five unmeasured warmup samples
followed by the required 30 paired samples. Changing a case, sample count, or
mock response shape requires review as a performance-contract change and keeps
the prior result for comparison.

The compile-time goal is also tested structurally. Editing one plugin crate MUST
NOT rebuild or relink the `aiperf` host or an unrelated plugin. A minimal host
build MUST NOT include gRPC, WebSocket, Dynamo, Parquet, OTLP, MLflow, or W&B
implementation dependencies. CI records clean build and isolated incremental
rebuild/link times; the migration is incomplete until the report demonstrates
dependency isolation even if wall-clock improvement varies by machine cache.

## Failure and trust policy

Stale third-party binaries are expected after an ABI-changing AIPerf upgrade.
Invalid or incompatible auto-discovered optional plugins are quarantined
without loading and remain visible through plugin inspection diagnostics.
Unrelated benchmarks remain usable. A first-party plugin required by the
installed distribution, or a manifest explicitly required by the user, fails
startup when missing or incompatible. Selecting a component available only
from quarantine fails validation with its stored cause.

Once library code is invoked, a missing symbol, manifest mismatch, registration
error, or registration panic rejects the complete package. Transactional
staging prevents partial registry visibility. A benchmark does not proceed
after a selected package fails composition. Runtime callback failures use the
normal typed run-failure paths and cannot mutate or replace the frozen catalog.

Native plugins are trusted code. They can access process memory and host
resources, perform system calls, abort, or violate memory safety through unsafe
Rust. Digests establish identity and reproducibility, not isolation.

Reports record the resolved plugin-lock digest. A detailed artifact records
loaded, winning, shadowed, quarantined, and required package identities without
including environment-sensitive absolute discovery paths.

The failure policy is fixed by phase and authority:

| Condition | Auto-discovered optional package | Distribution-required package | Explicit `--plugin-manifest` |
|---|---|---|---|
| Missing default discovery directory | Ignore | Fail missing requirement | N/A |
| Unreadable/invalid manifest | Quarantine and report | Fail composition | Fail command |
| Missing/wrong library type | Quarantine and report | Fail composition | Fail command |
| Digest or ABI mismatch | Quarantine without loading | Fail composition | Fail command |
| Restricted dependency resolution failure | Quarantine | Fail composition | Fail command |
| Equal-priority entry tie | Mark that key ambiguous | Fail if required key is affected | Fail if explicit package entry is affected |
| Entry symbol or descriptor mismatch after code load | Poison process and fail composition | Fail composition | Fail command |
| Registration error or panic | Roll back package, poison process, and fail composition | Fail composition | Fail command |
| Selected key is quarantined/ambiguous | Fail validation before effects | Fail validation before effects | Fail validation before effects |
| Runtime trait method returns error | Typed operation failure | Typed operation failure | Typed operation failure |
| Abort, UB, or memory corruption | Process may terminate | Process may terminate | Process may terminate |

"Quarantine" means the library entry symbol is not called when rejection occurs
before loading, none of its factories enters the active registry, and the exact
cause remains queryable. Quarantine is fixed for the process. There is no retry
after files change on disk.

Diagnostics MUST include category/name when applicable, normalized package name
and version, source API version, the mismatching fingerprint field when known,
and a remediation telling the user to rebuild or install a matching plugin.
Normal diagnostics omit absolute paths and library bytes. Debug diagnostics may
show a redacted/canonical package location but never dump plugin configuration
that may contain credentials.

Priority is not a security boundary. A higher-priority third-party package can
replace a first-party implementation and then runs with full process authority.
Installers and documentation MUST make this explicit. The loader does not
download code, resolve network package repositories, or modify installations.

## Migration

1. Record clean-build, incremental-build, link-time, binary-size, runtime, CPU,
   and allocation baselines.
2. Extract the ABI-facing API and focused reusable core crates while current
   implementations remain statically registered.
3. Add manifest parsing, discovery, priority resolution, ABI fingerprinting,
   plugin locking, dynamic loading, and a separately built minimal test plugin.
4. Migrate cold-path exporters: basic formats first, then Parquet, OTLP, MLflow,
   and W&B independently.
5. Migrate endpoint families while preserving prepared worker-local behavior.
6. Migrate transports last: HTTP first, followed by gRPC, WebSocket, and Dynamo.
7. Remove static fallbacks only after catalog, artifact, behavior, and
   performance parity pass for every first-party plugin.

Each numbered migration step is independently shippable and leaves one
authoritative composition path. During steps 2 through 6, a component is either
declared static or dynamic in the distribution inventory; it is never
simultaneously registered through hidden static and dynamic paths. A temporary
test-only comparison binary may contain both under different IDs, but production
IDs remain unique.

The exporter migration order is `basic`, `parquet`, `otlp`, `mlflow`, then
`wandb`; telemetry exporters are separate artifacts and do not form one
"telemetry" dependency bundle. Endpoint packaging follows measured dependency
and coupling boundaries rather than forcing one library per dialect. Transport
migration is `http`, `grpc`, `websocket`, then `dynosim`; HTTP is the proof that
the common online hot path satisfies the performance contract.

The static fallback for a component is removed only when all of these are true:

1. its plugin builds independently through the supported SDK command;
2. its manifest and actual registration conform on all supported platforms;
3. existing behavior and artifact suites pass unchanged or with an explicitly
   approved public-contract migration;
4. parent, child, and cellular lock agreement pass;
5. the normative performance gates pass;
6. editing the plugin does not rebuild/relink the host or unrelated plugins;
7. packaging installs and uninstalls the manifest/library pair atomically;
8. user-facing missing, incompatible, and override diagnostics are covered.

No migration step may claim compile-time success by disabling the compiler cache,
changing the optimized profile, weakening LTO in the static baseline, omitting a
required feature from the comparison, or moving work to an unmeasured build
script.

This is a selective reversal of the runtime monocrate consolidation, not a
return to many tightly coupled crates. A boundary is extracted only when it is
publicly reusable or produces a measured build-isolation benefit.

## Verification

- Strict manifest-schema, normalization, and path-hardening tests.
- Library digest, ABI fingerprint, compiler, target, and panic-strategy mismatch
  fixtures.
- Priority winner, shadowing, ambiguity, quarantine, and package-transaction
  tests.
- Declared-versus-actual registration conformance tests.
- Trait-object destruction and process-lifetime library-residency tests.
- A third-party exemplar built from a separate Cargo workspace.
- Parent/child re-execution and cellular plugin-lock agreement tests.
- Linux `.so`, macOS `.dylib`, and Windows `.dll` CI coverage.
- Existing protocol-v2, endpoint, transport, exporter, and mock-server suites.
- Static-baseline versus dynamic-plugin microbenchmarks and end-to-end benchmark
  comparisons.
- Clean and incremental build-time measurements for the host and each
  first-party dependency island.

The SDK ships a minimal plugin example, manifest generator, ABI inspection
command, and local conformance harness so third parties can detect incompatibility
before installation.

### Required conformance fixtures

The separately built exemplar suite contains at least:

- one library registering one endpoint;
- one library registering a transport and endpoint together;
- one library registering multiple exporters;
- one library with both winning and shadowed entries;
- one package that fails after an earlier staged registration, proving rollback;
- one package whose declaration disagrees with its manifest;
- one stale compiler/API fingerprint package that is never called;
- one equal-priority ambiguity across separate packages;
- cross-boundary `String`, `Vec`, `Box`, `Arc`, `Rc`, error, trait-object, and
  boxed-future allocate/return/drop coverage for every family retained by the
  final public traits;
- a library whose destructor records completion, proving process-lifetime
  residency and object-before-code teardown assumptions without production
  unload;
- same-host re-exec and remote-cell lock mismatch fixtures.

Tests MUST use the real dynamic loader and separately produced artifacts. A test
that constructs an extension directly in the test executable proves registry
behavior but does not satisfy loader, linkage, allocator, or ABI conformance.

### Documentation and tooling deliverables

The feature is not complete without:

- generated JSON Schema for `plugins.yaml` schema `2.0`;
- `aiperf plugins list`, `validate`, and `inspect-abi` documentation;
- a third-party Cargo template using only allowlisted public crates;
- platform installation layouts and atomic package install/uninstall guidance;
- an ABI mismatch troubleshooting guide;
- a priority/override security warning;
- a compatibility table distinguishing source API and exact ABI fingerprint;
- benchmark methodology and retained baseline results;
- an updated architecture index and extension-registry record that remove the
  old claim that native AIPerf has no dynamic discovery or dynamic-library seam.
