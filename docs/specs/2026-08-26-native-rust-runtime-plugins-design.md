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
   traits and Rust values declared by the AIPerf plugin API. No plugin entry,
   category-trait, or transferred-data boundary uses a C ABI, `abi_stable`
   facade, serialization layer, generated function table, Python runtime, or
   process RPC. This does not prohibit ordinary implementation dependencies,
   including the shared allocator provider, from using their upstream C ABI
   internally without becoming the plugin contract.
2. **Exact build compatibility.** A library is callable only when its
   SDK-produced `host_abi_universe_id` equals the host's identity, its unique
   `plugin_artifact_build_id` validates against that universe, and the complete
   distribution-controlled non-system executable artifact closure loaded by
   the platform loader is the exact closure acquired, hashed, and locked by the
   host. These identities are compatibility preflights, not compiler proofs
   that native Rust ABI is sound. Source compatibility never waives a binary
   mismatch.
3. **Composition before host effects.** Discovery, side-effect-free closure
   validation, priority resolution, native activation, transactional
   registration, and freezing complete before AIPerf creates any host-owned
   clock, Tokio runtime, worker, network client, artifact directory, cell, or
   benchmark effect. Native loading and plugin registration are themselves
   trusted initialization effects and cannot be made side-effect-free.
4. **One frozen universe.** Capabilities, validation, execution, re-executed
   children, controllers, and cells use one resolved plugin-lock identity. No
   execution path constructs a separate built-in registry or discovers an
   additional plugin.
5. **Process-lifetime residency after successful mapping.** Every library for
   which the platform loader returns a handle is retained before AIPerf resolves
   or calls its entry or permits any pointer/value to escape, and is never
   unloaded or replaced before process exit. A platform may itself unmap a
   module whose initializer fails before returning a handle; that failure
   poisons composition and no plugin value can have escaped. No trait object's
   vtable or future may outlive its defining code.
6. **No runtime mutation.** Registration is impossible after freeze. There is no
   reload, unload, enable, disable, override, or priority change during a run.
7. **Transactional packages.** A package contributes all declared entries or
   none. A registration error cannot expose a prefix of the package.
8. **Deterministic override.** Priority is resolved per normalized
   `(category, name)`. Filesystem order, discovery-source order, and load order
   never break a tie.
9. **First-/third-party parity.** First-party endpoints, transports, and
   exporters use the same manifest, native Rust boundary, host ABI universe,
   entry-symbol, registration, priority, and freeze mechanisms as third-party
   implementations. The host has no hidden static preference path.
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
16. **One allocator; no cross-boundary unwind.** Every artifact that can
    allocate or free a boundary-owned value resolves `Global` operations to one
    exact process allocator provider. No panic may unwind across a host/plugin
    call in either direction; the shipped host and every loadable plugin use
    `panic=abort`, so a panic is process-fatal rather than a recoverable plugin
    error.

### Decision traceability

This table is the authoritative record of the design decisions made during
review. Later edits MUST update both the detailed section and this table when a
decision changes.

| Decision | Normative resolution | Detailed section |
|---|---|---|
| Artifact form | Multiple Rust `cdylib` artifacts named `.so`/`.dylib`/`.dll`, exporting a native-Rust entry function | Library contract; rejected alternatives |
| Programming model | Native Rust traits and types; no C ABI or ABI-wrapper facade at the plugin entry/category/data boundary; implementation dependencies may retain their upstream ABI | Invariant 1; rejected alternatives |
| Compatibility meaning | Source API is SemVer; binary loading requires an exact common host ABI universe plus a validated package-specific artifact-build record; either identity changing requires the affected rebuild; neither identity is an ABI proof | Compatibility contract |
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
| Boundary memory and panic | One verified allocator provider owns all `Global` storage; shipped host/plugins use `panic=abort`; no unwind crosses the boundary | Rust linkage, allocator, and panic rules |
| Artifact acquisition | Immutable content-addressed generations bind the complete distribution-controlled non-system executable dependency closure; pathname hash-then-load is forbidden | Manifest format; composition and lifecycle |
| Alias precedence | Canonical IDs are resolved first and cannot be replaced by aliases; alias-only conflicts use priority | Discovery and priority |
| Config compatibility | Config v2 accepts either legacy or open forms, never mixed forms, and normalizes both before protocol projection | Runtime category behavior |
| Exporter capture | Exporter factories declare host-owned capture requirements; plugins never receive per-record callbacks solely for export | Exporters; performance contract |

### Invariant enforcement map

This map is normative. An invariant is not satisfied by prose intent alone; all
listed enforcement and evidence MUST exist before generation 1 is released.

| # | Enforcement | Required evidence |
|---|---|---|
| 1 | Rust-ABI entry and ordinary native Rust category traits; no C ABI/facade/serialization layer | Separately built `cdylib` exemplars and four-target real-loader trait calls |
| 2 | Embedded common-universe and package-build records plus immutable, staged, rehashed distribution-controlled non-system executable closure | Build-input mismatch, artifact-swap, dependency-tamper, and loaded-module identity fixtures |
| 3 | Frozen composition invoked before every named host-owned effect; plugin initialization explicitly remains trusted/effectful | Help/list/config/profile/eval/re-exec/cell effect-order tests |
| 4 | Full canonical lock carried by dedicated re-exec/cell bootstrap and bound into signed cell registration | Same-lock reproduction and first-difference mismatch tests for every process role |
| 5 | Every successfully returned handle enters the process-resident set before symbol resolution/pointer escape; Unix retains/pins and Windows pins modules; initializer failure before a returned handle poisons with no escape | Subprocess lifetime, failed-initializer, object-drop, and attempted-unload fixtures |
| 6 | Type-state builders expose no registration after consuming freeze; process-global differing-lock reuse fails | Compile-fail API tests plus same-process lifecycle tests |
| 7 | Package-scoped staging commits all declared registrations together | Multi-entry rollback and descriptor-disagreement fixtures |
| 8 | Versioned normalization, canonical-first alias rules, unique-max priority, deterministic ties | Canonical/alias/version/tie/shadow fixture matrix |
| 9 | Authenticated distribution inventory uses the same package path; migrated production IDs have no static path | Production-code searches and first-/third-party behavior parity suites |
| 10 | Checked crate allowlist and rustdoc-derived method/type/owner table forbid orchestration objects | Dependency-policy CI and separately built third-party workspace |
| 11 | No additional hot-path operation; exporter capture is host-planned and transport-neutral | IR/call-graph plus lock/allocation/dispatch instrumentation artifacts |
| 12 | Static baseline is retained until simultaneous one-sided non-inferiority and allocation gates pass | Frozen paired benchmark inventory, raw samples, and reproducible bootstrap report |
| 13 | Open registry IDs and strict plugin-owned config; Config-v2 legacy/open union rejects mixtures | Schema, projection, serialization, CLI, protocol, and unknown-ID tests |
| 14 | Root ownership/mode policy, privileged-mode restrictions, authenticated first-party inventory, explicit no-sandbox diagnostics | Discovery authority, privilege, tamper, rollback, and revocation tests |
| 15 | Intended catalog fixes winners before native activation; any later failure poisons or fails without re-resolution | Loader/registration/runtime failure and no-promotion fixtures |
| 16 | One verified `aiperf_alloc_v1` provider, with every `GlobalAlloc` shim importing and resolving its pinned `mi_*` ABI directly, plus `panic=abort` across all boundary artifacts | Cross-direction ownership/import-map tests, runtime GOT/bound-symbol/IAT origin checks, preload/interposition fixtures, and subprocess abort tests |

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
the request lifecycle. A post-run exporter could technically run out of process
without hot-path IPC, but that would give exporters a different first-/third-
party lifecycle, report-ownership, failure, and packaging mechanism. That hybrid
was rejected to preserve one native plugin model; it is not rejected on exporter
throughput grounds. Delegating a complete benchmark to one process avoids
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

### Rust `dylib`

Crate type and function ABI are independent: both `dylib` and `cdylib` can
export an unmangled function using the native Rust ABI. The cross-platform
experiment at `https://github.com/ajcasagrande/rust-native-plugin-lab` loads the
same native Rust trait object from both forms on Linux x86-64, macOS arm64,
Windows x86-64, and Windows arm64. On the pinned Linux build, `dylib` exported
1,764 dynamic symbols and was 1.5 MiB while `cdylib` exported only the entry
symbol and was 397 KiB; both had the same system-library dependencies.

Those numerical observations are pinned to lab commit
`3f55f62ed1ca67fb15b50c5c316c98633fdfb656`, GitHub Actions run
`32950193501`, rustc `1.97.1 (8bab26f4f 2026-07-14)`, target
`x86_64-unknown-linux-gnu`, and `cargo build --workspace --release`. The
artifact report used `readelf -d`, `nm -D --defined-only`, and byte sizes from
the resulting release artifacts. A later branch or moving repository URL is not
evidence for those exact numbers.

Generation 1 therefore uses `cdylib` as the tighter final plugin artifact.
`cdylib` does **not** make the entry function `extern "C"`; the SDK explicitly
types it as a Rust-ABI function. Rust `dylib` is rejected because its rustc
metadata and broad Rust-symbol export surface are unnecessary for handle-scoped
runtime lookup and did not produce a shared `std`/API topology in the experiment.
The observed success proves mechanical feasibility only, not ABI soundness,
allocator compatibility, panic safety, or independent-build compatibility.

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

### Lockfile-first-only execution

Requiring users to generate and review a lock before every normal invocation
would improve reproducibility but would not provide the requested install-and-
auto-discover experience. It is rejected as the only production mode. The host
still produces a complete canonical lock on every composition, accepts an
explicit `--plugin-lock` for hermetic operation, and requires children and cells
to reproduce that lock exactly.

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

### Per-library allocators with origin-specific destroy functions

Letting each plugin allocate with its own allocator and returning every owned
value through an origin-specific `destroy`/`free` callback can be made correct,
but it changes ordinary Rust ownership into a generated destruction protocol.
It adds function-table calls and origin metadata to drops, makes `String`,
`Vec`, `Box`, `Arc`, `Rc`, errors, trait objects, and futures non-native at the
boundary, and moves work onto request completion and potentially token-heavy
paths. It was rejected because it violates both the native-trait requirement and
the zero-added-hot-path-operation invariant.

### Forbid all owned Rust values across the boundary

A borrowed/static-only API would avoid cross-library allocation ownership, but
endpoint, transport, async, error, and exporter contracts inevitably need owned
buffers, futures, handles, and results. Requiring plugins to retain every value
behind bespoke borrowed handles would recreate an ABI facade and complicate
lifetime management. It is permitted only as the limited feasibility-spike
stage before allocator conformance, not as the production contract.

### Switch the complete process to `System`

Using Rust's `System` allocator everywhere could reduce provider packaging on
some Unix systems, but it abandons AIPerf's existing mimalloc performance
baseline, does not by itself prove one compatible allocator instance across
every Windows/native dependency topology, and would be a product-wide allocator
migration unrelated to plugin modularity. It was rejected. The dynamic mimalloc
provider must instead prove non-inferiority against the current statically linked
mimalloc baseline.

### AIPerf allocator wrapper or allocation function table

An `aiperf_alloc_v1` Rust wrapper, selector, or generated table would provide an
obvious stable ownership seam, but every allocation/reallocation/free would add
another AIPerf-authored call or indirect dispatch and would no longer be the
existing allocator path. It was rejected. `aiperf_alloc_v1` names and pins the
shared mimalloc binary; the Rust `GlobalAlloc` shims import upstream `mi_*`
symbols directly.

### Unwind containment at the plugin boundary

Keeping `panic=unwind` and wrapping every boundary call with `catch_unwind`
would add containment calls, cannot make all foreign/native unwind interactions
sound, and conflicts with the zero-additional-call-layer requirement. It was
rejected in favor of typed recoverable errors and `panic=abort`. This changes the
current execute-mode panic envelope and therefore requires the explicit product
migration described below.

## Public feasibility lab: evidence and gate disposition

The standalone public lab at
`https://github.com/ajcasagrande/rust-native-plugin-lab` is intentionally not an
AIPerf codebase. The authoritative advanced result is commit
`f5af252970ef73031da6fe0c449a894e0cc3b4ea` and GitHub Actions run
`32955454915`. That run is green on Linux x86-64, macOS arm64, Windows x86-64,
and Windows arm64. It builds real Rust `dylib` and `cdylib` plugins, one shared
mimalloc 3.5.0 provider, host/mismatch/panic fixtures, and same-source shared and
static allocator benchmarks. Each target uploads eight binary-evidence
artifacts with format, architecture, exact size, imports, and exports.

The exact-build mechanical results are positive:

- plugin-allocated `String`, `Vec<u8>`, `Box<u64>`, `Arc<String>`, `Rc<String>`,
  and `Pin<Box<dyn Future<Output = String>>>` cross to the host and are dropped
  there; host-created owned values cross and are consumed/dropped by the plugin;
- a stale revision whose every trait method aborts is rejected before the first
  trait call, a valid `panic=abort` plugin terminates only its test subprocess,
  and successfully returned library handles remain resident;
- `aiperf_alloc_v1` is modeled as the shared mimalloc binary itself, not an
  AIPerf wrapper/table. Host, `dylib`, and `cdylib` directly import all nine
  required symbols: `mi_free`, `mi_malloc`, `mi_malloc_aligned`, `mi_realloc`,
  `mi_realloc_aligned`, `mi_subproc_main`, `mi_version`, `mi_zalloc`, and
  `mi_zalloc_aligned`; and
- ordinary, zeroed, aligned, reallocation-prefix-preservation, and free paths
  pass. Host and plugin observe mimalloc version `30500` and the same
  process-global `mi_subproc_main()` pointer. Windows arm64 evidence is
  PE/AArch64 and imports the required functions from `mimalloc.dll`.

The allocator performance result is a **risk signal, not a passed performance
gate**. One five-million-iteration CI smoke run produced these raw shared versus
same-source-static mimalloc 3.5.0 timings:

| Target | `mi_version`, shared vs static | 64-byte aligned alloc/free, shared vs static |
|---|---:|---:|
| Linux x86-64 | 6.319 vs 4.910 ns/op (`+1.409`) | 8.619 vs 6.344 ns/op (`+2.275`) |
| macOS arm64 | 2.875 vs 2.277 ns/op (`+0.598`) | 6.045 vs 5.766 ns/op (`+0.279`) |
| Windows x86-64 | 5.483 vs 4.387 ns/op (`+1.096`) | 8.433 vs 8.072 ns/op (`+0.361`) |
| Windows arm64 | 1.942 vs 1.480 ns/op (`+0.462`) | 7.047 vs 6.909 ns/op (`+0.138`) |

These single-run measurements conflate dynamic import/PLT/IAT cost, optimizer
and linker differences, allocator work, and hosted-runner noise. They are not a
production regression estimate and cannot be used to waive or fail the
normative paired protocol. They do establish that allocator linkage overhead is
measurable enough to require the full gate. The shared allocator topology is
therefore mechanically feasible, but the zero-loss invariant is **not yet
proven**. Any failure of the frozen structural, allocation-count/byte, or
simultaneous non-inferiority gates rejects this architecture; production may
not accept the measured overhead as a tradeoff.

The lab also confirms limits that remain fundamental:

1. Exact-build success does not establish a stable or sound Rust ABI for native
   calls, trait objects, standard-library owned types, `Arc`/`Rc`, or futures.
2. A revision check runs only after the native entry value has been called and
   decoded; a calling-convention or entry-layout mismatch can be undefined
   behavior before the check.
3. Shared allocation is correct only while every relevant artifact actually
   resolves to the same permanently resident provider; replacement, unload,
   search-path substitution, or symbol interposition is unsafe.
4. Panics cannot cross this seam and therefore abort the process.
5. Residency is intentional process-lifetime retention, not safe unload.
6. An unmangled export fixes symbol spelling only, not Rust ABI.
7. The smoke benchmark cannot isolate pure import-stub cost.

The lab once produced false-green linkage evidence because Mach-O leading
underscores were not normalized and piped PowerShell native-command exit codes
were not propagated. Commit `f5af252` adds red/green regression tests and fails
closed on incomplete evidence. Production evidence tooling MUST preserve those
lessons: normalize platform symbol spellings explicitly and propagate every
native inspector's real exit status through pipelines.

## Compatibility contract

The public plugin API is source-stable within its declared compatibility policy,
but the binary ABI is exact-build only. Rust does not stabilize the ABI of trait
objects, `repr(Rust)` layout, standard-library containers, async futures, enums,
or unwinding across independently linked artifacts. Rust exposes no compiler-
authenticated whole-crate ABI hash. Consequently, this design does not claim to
prove ABI equality.

The SDK emits two distinct records because common host/plugin ABI facts and a
plugin's private build facts cannot truthfully have one byte-identical value:

- `host_abi_universe_id` is the BLAKE3 digest of the exact compatibility facts
  that the host and every plugin MUST share. A library is rejected unless this
  ID is byte-identical to the host's ID.
- `plugin_artifact_build_record` is the unique canonical record of one plugin
  artifact and its hermetic build. Its BLAKE3 digest is the
  `plugin_artifact_build_id`. The record declares the common
  `host_abi_universe_id` against which it was built, but it is not expected to
  equal either the host record or another plugin's build record.

The SDK computes both rather than asking authors to maintain them. A change to
an AIPerf boundary, compiler, sysroot, ABI-facing artifact, allocator-provider
contract, panic policy, or boundary linker topology produces a new
`host_abi_universe_id` and requires every plugin to rebuild. A change confined
to one plugin's private source, dependency, build script, or non-system native
closure changes only that plugin's `plugin_artifact_build_id`. It does not
rebuild the host or an unrelated plugin. A human-readable source API version
remains separate so diagnostics distinguish a deliberate API revision, a
common-universe mismatch, and a package-build mismatch.

"Stable" in this design means only that the published Rust source API follows
AIPerf's compatibility policy. It does **not** mean that one compiled library
survives a relevant identity change. Documentation and diagnostics MUST use the
terms "source API version", "host ABI universe identity", and "plugin artifact
build identity" separately and MUST NOT advertise a stable or proven Rust
binary ABI.

The plugin source API has its own SemVer version beginning at `1.0.0`, independent
of the AIPerf product version. A major change may remove or alter public source
contracts. A minor change is additive and retains source compatibility for
conforming plugin crates. A patch change corrects behavior without changing the
documented source signature. This SemVer promise affects recompilation only:
even an additive or patch release may produce a different host ABI universe,
in which case installed binaries MUST be rebuilt.

Both canonical records are length-delimited and sorted. IDs are BLAKE3 digests
of the canonical bytes. The `host_abi_universe_id` record contains only facts
that can be identical between the host and every conforming plugin:

```text
plugin_api_generation
source_api_version
rustc_commit_hash
rustc_executable_digest
rustc_full_version
sysroot_artifact_logical_id -> digest
proc_macro_binary_logical_id -> digest
target_triple
target_specification_digest
target_pointer_width
target_endian
panic_strategy
codegen_backend
boundary-affecting normalized --cfg and -C/-Z flags
abi_facing_compiled_crate_artifact -> digest
allocator_provider_contract_version -> loader identity and artifact digest
boundary_linker_topology_policy
target_system_library_policy_version
```

The `plugin_artifact_build_record` contains:

```text
host_abi_universe_id
plugin_package_name_and_version
plugin_source_and_declared_feature_input -> digest
complete normalized rustc invocation -> digest
complete normalized linker invocation -> digest
all effective --cfg and -C/-Z flags
build_script_executable -> digest
hermetic_build_input_and_normalized_environment -> digest
build_script_output_and_generated_source -> digest
plugin_link_payload_before_record_embedding -> digest
plugin_main_artifact_loader_identity
distribution_controlled_non_system_dependency -> digest, loader identity, and edges
system_library_policy_version
observed_system_library_identity -> version/build identity when available
```

Canonical compatibility records contain logical labels, normalized values, and
digests, never host-specific absolute paths. Path-bearing compiler and linker
arguments are path-remapped before canonicalization. The raw invocations and
local paths MAY be retained in a separate diagnostic artifact, but they do not
participate directly in a reproducible identity.

The common ABI-facing compiled-crate closure is conservative but not allowed to
swallow plugin-private implementation code. It includes every exact prebuilt
artifact whose representation, validity rule, trait/vtable layout, auto-trait
contract, associated boundary type, generic boundary wrapper, drop convention,
panic behavior, or allocator behavior is instantiated, interpreted, or relied
on by both host and plugin. For a shared concrete boundary type this includes
layout-affecting private fields and generated code even when absent from its
public signature. A dependency is common unless the SDK proves that only one
side can instantiate or interpret its values/code. Cargo metadata and source-
tree hashes alone are insufficient.

By contrast, a plugin's erased concrete factory, endpoint, transport, exporter,
future, error, or validated-config type; its method bodies and concrete vtable/
drop glue; and dependencies reachable only inside that implementation remain in
that plugin's artifact-build record. The common universe binds the shared trait
object representation, method signatures, boundary wrappers, and destruction
calling convention, not the byte identity of each plugin-owned implementation.
The host may call plugin vtable methods/drop glue only while its library is
resident, but it never instantiates or interprets the erased concrete layout.
Changing such private code therefore rebuilds only that plugin. If a private
type becomes concrete at the boundary or host code begins to instantiate/
interpret it, the SDK moves its exact artifact closure into the common universe
and all plugins rebuild.

The SDK embeds both records and IDs in a non-executable, platform-specific
artifact section and repeats their IDs in the manifest. The loader parses and
compares that section without invoking the platform dynamic loader. A mismatch
between the sidecar, manifest, and embedded records quarantines the artifact
before native activation. Unknown record fields are rejected by an older host
rather than ignored. `aiperf plugins inspect-build` reports the first differing
common-universe or package-build field. The supported SDK supplies byte-
identical ABI-facing compiled artifacts; independently recompiling allegedly
identical sources is not a conforming build.

The package build ID never hashes a byte string containing that same ID. The
build record binds the deterministic linked payload before record embedding;
the manifest and canonical plugin lock separately bind the digest of the final
whole artifact after embedding and any required platform signing. Verification
checks both relationships. An implementation MAY equivalently define a
versioned normalized-artifact digest that zeros the complete embedded-record
section, but raw self-referential whole-file hashing is forbidden.

## Crate architecture

The current runtime monocrate will be split only at measured plugin-facing
boundaries:

```text
aiperf-plugin-api
|-- AIPerfExtension and registration contracts
|-- endpoint, transport, and exporter factory traits
|-- descriptors, registry identifiers, and host ABI universe identity
`-- native types that cross the library boundary

aiperf-core
|-- request and response models
|-- clocks, dispatch, observation, and measurement contracts
|-- endpoint body and response-reduction helpers
`-- report and metric types

aiperf-plugin-sdk
|-- plugin declaration and entry-symbol macros
|-- manifest generation and validation support
|-- host-universe, package-build, and artifact-closure inspection helpers
`-- plugin conformance test harness

aiperf-endpoint-sdk
aiperf-transport-sdk
aiperf-export-sdk
`-- category-specific helpers with isolated dependency surfaces
```

The host runtime depends downward on the API and core crates. Plugin crates
depend on the API plus only the core or category SDK crates they need. Plugins
do not depend on the complete orchestration runtime. This avoids dependency
cycles, duplicate process-global ownership, and needless host-universe
expansion.

`aiperf-plugin-api` MUST NOT depend on a transport framework, exporter backend,
CLI parser, Tokio runtime, or the orchestration crate. CI enforces a checked
dependency allowlist; comparative language such as "smallest crate" is not a
conformance test. ABI-facing model types that are currently defined inside large
modules move either into this crate or a narrowly scoped core crate. A crate is
ABI-facing if a boundary method accepts, returns, lays out, drops, or creates a
value involving that crate, including behind `Box`, `Rc`, `Arc`, `Result`, a
future, a private field, or another container.

The workspace MUST publish an explicit API allowlist. A third-party plugin is
supported when it imports only allowlisted AIPerf crates and public items.
Runtime-private crates remain unpublished or expose no plugin-facing public
surface. Adding a crate to the allowlist is an API decision and requires a
host-universe input; merely making a Rust item `pub` does not make it supported.

Plugins MAY use arbitrary ordinary Rust dependencies internally. Such a
dependency enters the common host ABI universe when one of its types, traits,
panics, allocator behaviors, or native handles crosses the host/plugin boundary.
Pure implementation dependencies remain in that plugin's artifact-build record
and can change without rebuilding the host or unrelated plugins.

The ABI-facing compiled-artifact set is generated, reviewed, and checked into
the SDK release. It is the conservative closure defined by the compatibility
contract, not merely crates named in public signatures. Build scripts cannot
remove an ABI-facing artifact. When review finds an untracked input, the SDK
release is invalid and artifacts produced by that record are revoked.

Before API generation 1 is published, `docs/specs/plugin-api-ownership.md` MUST
list every public boundary trait method, its owning crate, each argument and
return type, allocation owner, drop owner, panic policy, and whether the method
is startup-only or hot-path. CI parses rustdoc JSON and fails when the exported
API differs from that table. Today's `RunContext`, closed `Transport`, concrete
HTTP sink config, complete `AIPerfRegistry`, CLI config, and orchestration engine
objects are forbidden at the boundary. They are replaced by narrow transport-
neutral handles for clock, observation, artifact policy, endpoint preparation,
and execution services.

Process-owned facilities remain host-owned and are passed into plugins as
native trait handles. These include clocks, scheduling, observation, artifact
policy, and worker construction context. A plugin may use those services but
must not create a competing process-global instance.

### Rust linkage, allocator, type identity, and panic rules

The SDK controls the complete plugin build, not just a header crate. A
conforming artifact MUST:

- use Cargo `cdylib`; the entry function remains native Rust ABI;
- use the exact compiler and target declared by the host SDK bundle;
- consume the exact prebuilt ABI-facing crate artifacts supplied by the SDK;
- use `panic=abort` for every profile that can produce a loadable artifact;
- use the SDK-injected global allocator shim and no author-selected allocator;
- not export a second copy of an AIPerf process-global singleton;
- include actual link/import/export maps and loaded-module expectations that the
  manifest generator validates; and
- export only the plugin entry symbol intentionally exposed by the SDK.

The executable's existing `#[global_allocator]` declaration does not govern a
separately linked plugin. Generation 1 therefore names one distribution-owned
shared allocator provider identity, `aiperf_alloc_v1`. This is the pinned shared
mimalloc binary itself, **not** a new Rust wrapper, function table, selector, or
dispatch service. The host's existing Rust `GlobalAlloc` shim and the SDK-
injected plugin `GlobalAlloc` shim directly import the provider's `mi_malloc`,
`mi_zalloc`/calloc, `mi_realloc`, aligned-allocation, and `mi_free` exports. No
intermediate AIPerf allocation function is permitted. The provider's digest,
loader identity, and import contract are part of the common host ABI universe
and every plugin build record. The provider is a distribution-baseline module
loaded as an operating-system dependency of the executable; it is not copied
into or staged separately for each plugin package. At the first instruction
under AIPerf control, before discovery or plugin preflight, the process
enumerates the already-loaded provider, opens the exact mapped object without
causing a second load, and verifies its loader identity and digest against the
authenticated distribution baseline. Absence, ambiguity, or mismatch aborts
startup. Plugins reference this verified baseline requirement by `(loader
identity, digest)`.

The executable carries a mandatory, non-delay-loaded dependency on the
provider: ELF `DT_NEEDED` plus distribution-owned `$ORIGIN` resolution; Mach-O
`LC_LOAD_DYLIB` with `@loader_path`; and a normal PE import descriptor resolved
from the protected application distribution directory, never a delay-load
descriptor or current-directory search. Thus the loader maps and may initialize
the provider before AIPerf can hash it. This unavoidable startup trust is
restricted to the authenticated, access-controlled distribution baseline: a
post-map digest mismatch is process-fatal and cannot be quarantined or undone.
No optional or third-party plugin module is mapped before that verification.

`aiperf_alloc_v1` deliberately uses mimalloc's existing upstream C calling ABI
for allocator implementation imports. The prohibition on a C ABI applies to the
plugin entry, category traits, and transferred values; it does not require
reimplementing a third-party allocator's established import ABI. There are no
C DTOs, serialization, function tables, or marshalling at the plugin category
boundary. Provider functions MUST be no-unwind and process-lifetime resident.

The injected `GlobalAlloc` shim has these exact semantics:

- a nonzero layout whose alignment is at most mimalloc's guaranteed natural
  alignment calls `mi_malloc`; larger valid power-of-two alignments call the
  pinned aligned-allocation export;
- zeroed allocation uses `mi_zalloc` or the corresponding pinned aligned-zeroed
  export; use of `calloc` is permitted only when its size multiplication is
  checked before the call;
- reallocation calls the matching pinned ordinary or aligned reallocation
  export, preserves the original allocation when null is returned, and exposes
  Rust's ordinary allocation-failure behavior to the caller;
- deallocation calls `mi_free`; Rust zero-sized allocation sentinels and null
  pointers are never passed to a provider deallocation operation; and
- the shim validates Rust `Layout` size/alignment preconditions and contains no
  lock, lazy selector, indirect function table, allocation metadata, or AIPerf-
  authored allocation wrapper between `GlobalAlloc` and `mi_*`.

Actual ELF/Mach-O imports and PE import tables MUST prove that each compiled
`GlobalAlloc` shim targets that provider; source configuration is not evidence.
Allocator bindings are never lazy: the executable and plugins carry ELF
`DF_BIND_NOW`/`DF_1_NOW` from `-z now`; Mach-O artifacts use bind-at-load/non-
lazy symbol pointers and contain no lazy-bind opcode or lazy pointer for any
required `mi_*` symbol; and PE uses the normal non-delay IAT. Static inspection
rejects a lazy allocator relocation even if the platform loader would likely
resolve it before first use. Plugin ELF loading additionally uses `RTLD_NOW` as
specified below.

After eager relocation and before any plugin entry call, the loader also resolves
each required `mi_*` symbol on the exact retained provider handle and proves
that every host/plugin shim relocation target equals an address inside that
provider's mapped executable ranges. ELF GOT/relocation slots, Mach-O bound
symbol pointers, and PE IAT entries are inspected through platform-specific
loader APIs. A symbol-name/import-table match with a different resolved address
is interposition and poisons startup. The host shim is checked immediately after
baseline verification; each plugin shim is checked immediately after mapping
but before calling its entry symbol. Conformance includes preload/interposition
fixtures on every platform.
Import maps cannot prove the origin of every explicit allocation in arbitrary
native code. Boundary-owned storage therefore MUST NOT be allocated through
`System`, direct libc allocation, a native library allocator, or another
explicit allocator unless its final destruction remains origin-matched wholly
inside the same library. A package is non-conforming if it contains or imports
another allocator for boundary-owned storage.

Windows Rust allocation shims link directly through the pinned mimalloc import
library; generation 1 does not depend on process-wide UCRT patching or
`mimalloc-redirect.dll` for Rust boundary storage. Native third-party libraries
may use their own allocator only for storage whose allocation and final free both
remain inside that library and never enter a boundary-owned Rust container.

Allocator option initialization moves into the distribution provider itself,
because an executable-owned hook cannot reliably configure an already-loaded
provider before that provider's constructor. In particular, the current Linux
priority-100 `.init_array` hook and generated
`aiperf_mi_option_arena_eager_commit()` ordering before mimalloc's priority-101
constructor are removed only after the provider reproduces the same eager-
commit option before its own allocator initialization. Equivalent macOS and
Windows initialization is provider-owned. Conformance verifies option value,
constructor order, first-allocation behavior, committed/resident memory, and
startup/runtime performance on every target; import topology alone is
insufficient evidence.

Until this shared-provider topology passes the cross-platform conformance gate,
no boundary trait may transfer ownership of `Global` storage, including
`String`, `Vec`, `Box`, `Arc`, `Rc`, owned errors, or boxed futures. Borrowed
descriptors and plugin-owned static factory objects may cross during the spike,
but no first-party migration may begin. After the gate passes, the public API
may include only container families named in the ownership table and exercised
by allocate/reallocate/return/drop tests in both directions. Plugin-created
trait objects are destroyed through their plugin vtables while the code and
allocator provider remain resident.

The host never performs `Any` downcasts on a value defined by a plugin. A
validated configuration is opaque to the host and is passed only to the exact
factory instance that created it; a downcast wholly inside that defining plugin
is permitted. Cross-artifact `Any`/`TypeId` checks are forbidden in generation
1, even when a conformance test happens to pass.

No panic may unwind across a boundary. Shipped production host, plugin,
allocator, and ABI-facing runtime artifacts all use `panic=abort`; the host ABI
universe, plugin build record, and inspection of each final production artifact
enforce it. Cargo test binaries, benchmarks, build scripts, and proc macros are
inspected under their own profiles and are never treated as evidence for the
production artifact. Entry, registration, factory, runtime callback, host
service, and drop panics are process-fatal. `catch_unwind` is not used to
classify a plugin panic as a recoverable error. Expected failures return typed
`Result` values. This removes cross-runtime unwind from the contract and adds no
hot-path containment wrapper.

This is an explicit public failure-behavior migration for AIPerf's current
`--execute` panic-to-protocol-envelope handling: in a plugin-capable production
binary, an in-process panic aborts before that envelope can be authored. Any
requirement to preserve a terminal protocol envelope after panic MUST be
implemented by an outer supervisor process observing abnormal termination, not
by `catch_unwind` in the plugin-capable address space. The migration cannot ship
until the CLI/protocol documentation and tests identify which behavior is the
public contract and either approve abort semantics or introduce that supervisor.

### Supported author workflow

The supported third-party workflow is one command owned by the SDK, exposed as
`cargo aiperf-plugin build --release`. It obtains the host compatibility record
from an installed AIPerf SDK bundle or an explicit `--sdk` directory, verifies
the exact Rust toolchain, invokes Cargo with the required crate type and linker
policy, validates the distribution-controlled non-system executable closure,
allocator imports, panic policy, and
exported symbols, embeds the build record, emits the manifest, then hashes every
closure artifact into that manifest. Handwritten manifests and ordinary `cargo
build` outputs may be inspected, but they are not conforming distributable
artifacts until the SDK validator accepts them.

The supported command executes Cargo, rustc, linkers, build scripts, and proc
macros in an SDK-owned hermetic build sandbox: network access is disabled; the
readable filesystem is the closed, declared, content-addressed input set; the
environment is allowlisted and normalized; output is private; and time,
randomness, locale, working-directory, and path-remapping policy are fixed and
recorded. An undeclared read, write, environment access, network access, or
non-deterministic input fails the build. The canonical plugin build record binds
the admitted input set and produced outputs. Merely logging ambient build-script
inputs is not a conforming approximation of this closure.

The SDK bundle contains the allowlisted crate versions/sources, target identity,
toolchain file, exact compiled ABI-facing artifacts, allocator-provider
requirement, host ABI universe record, hermetic-build policy, platform linker
policy, and JSON Schema. It does not
include or expose orchestration-private crates. A
plugin can use `aiperf-core` and category SDK helpers as normal Rust dependencies
and receives ordinary compiler type checking and documentation for them.

## Library contract

Each plugin is a Rust `cdylib` and exports one plugin entry symbol named
`aiperf_plugin_entry_v1`. It is a Rust-ABI function, not an `extern "C"`
function. Ordinary linker exports are not additional plugin entry points, and
lookup is always scoped to the exact library handle:

```rust
pub type PluginEntryV1 = unsafe fn() -> PluginDeclarationV1;

pub struct PluginDeclarationV1 {
    pub package: &'static PluginPackageDescriptor,
    pub extension: &'static dyn AIPerfExtension,
}

pub trait AIPerfExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>)
        -> Result<(), ExtensionError>;
}
```

The exact fields of the two public structs belong to `aiperf-plugin-api`; the
sketch fixes the borrowed-static declaration and call shape. The SDK macro emits
`#[unsafe(export_name = "aiperf_plugin_entry_v1")] pub unsafe fn ...`, embeds the
package descriptor and build record, and prevents authors from selecting a
different symbol name. The loader calls the symbol only after sidecar, embedded
records, complete distribution-controlled non-system closure, and both build-
identity validations. Unsafe code in host loader and SDK-generated glue is
confined to immutable artifact acquisition, process-lifetime residency, handle-
scoped symbol resolution, and the initial native Rust call; this does not claim
that trusted third-party plugin implementations contain no author-owned unsafe
Rust.

`PluginPackageDescriptor` repeats package name, version, source API version,
`host_abi_universe_id`, and `plugin_artifact_build_id`. The loader MUST compare
every repeated field with the manifest after the entry call. A mismatch rejects
the entire package. This second check detects accidental manifest/library
pairing errors; it is not a sandbox against malicious native code.

`PluginRegistrar` is a manifest-bound facade over a private staged registry. It
supplies package identity and priority from the manifest, observes every actual
registration, and exposes only the endpoint, transport, and exporter categories
in this version. A plugin cannot directly mutate the aggregate registry or
claim a different origin.

The extension registers ordinary `EndpointFactory`, `TransportFactory`, and
`ExporterFactory` trait implementations. One library may register any number
and mix categories freely. Packaging granularity is consequently a build and
dependency decision, not an API constraint.

First-party package boundaries MUST follow the checked feature/dependency
ownership matrix rather than one library per implementation. Small basic
exporters share one library; Parquet, OTLP, MLflow, and W&B MUST remain four
separate distribution-controlled non-system executable closures because their
dependency and release surfaces
differ. CI fails if one telemetry package links another telemetry backend's
implementation dependency. HTTP, gRPC, WebSocket, and Dynosim grouping follows
the normative feature matrix and measured coupling recorded during migration.

Every object created from plugin code is destroyed while its library remains
resident. The process-global `LoadedLibrarySet` intentionally retains library
handles until operating-system process teardown, including after an
`Application` value is dropped. `Application` owns the frozen registry and
ordinary plugin-created runtime objects; dropping them cannot unload code. This
is stronger than relying on Rust struct-field drop order.

## Manifest format

Each package publishes one `plugins.yaml` or package-qualified
`.plugins.yaml` that selects its immutable content-addressed generation:

```yaml
schema_version: "2.0"

plugin:
  name: aiperf-export-otlp
  version: "0.12.0"
  api_version: "1.0.0"
  host_abi_universe_id: "blake3:<host-abi-universe-id>"
  plugin_artifact_build_id: "blake3:<plugin-artifact-build-id>"
  main_artifact: store/<package-generation>/libaiperf_export_otlp.so

baseline_requirements:
  - role: allocator
    loader_identity: libaiperf_alloc_v1.so
    digest: "blake3:<allocator-digest>"

artifacts:
  - path: store/<package-generation>/libaiperf_export_otlp.so
    role: plugin
    digest: "blake3:<plugin-digest>"
    loader_identity: libaiperf_export_otlp.so
    dependencies: [baseline:allocator]

exporter:
  otel:
    priority: 0
    description: OpenTelemetry report exporter
    metadata: {}
```

Multiple packages in one discovery directory use package-qualified manifest
names while retaining the same schema, for example:

```text
plugins.d/
|-- aiperf-http.plugins.yaml
|-- store/<http-generation>/libaiperf_http.so
|-- aiperf-export-otlp.plugins.yaml
`-- store/<otlp-generation>/libaiperf_export_otlp.so
```

The macOS and Windows layouts substitute their platform library filenames. A
package-specific directory explicitly added with `--plugin-path` may use the
unqualified `plugins.yaml` name.

The manifest is declarative inventory. It does not name Rust types or select
individual symbols. The fixed package entry symbol performs registration. Each
artifact path is relative to the discovery-root handle; absolute paths, parent
traversal, alternate data streams, and path escaping are forbidden.

Manifest schema `2.0` is strict: unknown root, package, category, entry, or
metadata fields are rejected unless the relevant category schema explicitly
declares them. Required package fields are `name`, `version`, `api_version`,
`host_abi_universe_id`, `plugin_artifact_build_id`, and `main_artifact`.
`version` and `api_version` are
strict canonical SemVer strings; non-canonical equivalent spellings are
rejected. `artifacts` contains the main plugin and every non-system executable
native dependency. At least one supported category entry is required.

Package and entry names use normalization version 1: reject non-ASCII input;
trim ASCII space and tab at both ends; ASCII-lowercase; replace each `-` byte
with `_`; then require `^[a-z0-9][a-z0-9_]{0,127}$`. Empty results, other bytes,
consecutive authored separators, and unsupported normalization versions are
rejected. Authored spelling is display-only. The lock contains
`normalization_version: 1`. `baseline_requirements` references already-loaded
distribution modules by role, loader identity, and digest; these modules MUST
NOT also appear as package-local `artifacts`.

Native schema 2.0 intentionally shares the conventional `plugins.yaml` basename
with Python AIPerf's existing schema 1.0 entry-point packages, but the schemas
and runtimes are non-interoperable. Native discovery accepts only the exact
native root shape and `schema_version: "2.0"`; encountering the Python root or
schema 1.0 yields the stable diagnostic `python-plugin-manifest-not-native` with
guidance to use the Python installation path, not a generic corrupt-plugin
message. Native installers never publish into the Python entry-point resource
directory, and Python entry-point discovery never scans native `plugins.d`.

Each artifact record contains a relative path, BLAKE3 content digest, platform
loader identity (`SONAME`, Mach-O install name, or case-insensitive PE module
basename), role, and sorted dependency edges. Dependency edges are either a
package artifact loader identity or a typed `baseline:<role>` reference that
resolves exactly one `baseline_requirements` entry. The SDK target record defines a
versioned target system-library policy. Platform system libraries admitted by
that policy are outside the byte-exact artifact closure; the loader records
their observed runtime version/build identities when the platform exposes them.
Every distribution-controlled or third-party non-system transitive executable
dependency MUST appear in `artifacts`, except a verified distribution-baseline
module named in `baseline_requirements`. Unresolved ambient lookup is forbidden.
Two packages using the same loader identity MUST declare byte-identical content.
Otherwise every optional claimant of that identity is quarantined as one
conflict group before priority resolution; a distribution-required or explicit
claimant makes composition fail. No conflicting claimant is loaded or promoted.
Windows applies this rule case-insensitively to every non-system DLL basename.

Install and update use immutable content-addressed package-generation
directories. The installer writes and verifies a complete generation off-path,
then publishes one manifest generation with an atomic same-filesystem replace.
Readers acquire the manifest once, open the referenced generation relative to
the discovery-root directory handle, and never rescan it. Uninstall atomically
publishes absence first and garbage-collects unreachable generations later.
Windows defers deletion while a process holds a DLL. The distribution-required
inventory and every first-party reference publish through one equivalent atomic
generation so a process cannot observe a mixed first-party set.

Every manifest and artifact is acquired using no-follow, directory-relative
handles from the selected immutable generation. The host hashes the exact open
file objects. Because supported loaders cannot uniformly map an existing file
handle, the host copies the verified closure into a private host-owned, mode-
immutable, content-addressed staging generation, reopens and rehashes every
staged object, and loads only that staged absolute pathname. Pathname hash-then-
`dlopen`/`LoadLibraryEx` is non-conforming. Directory components, symlinks,
junctions, and Windows reparse points are checked relative to acquired handles.
Post-load metadata comparison is diagnostic only.

Every category entry contains `priority` as a signed 32-bit integer with default
zero, `description` with default empty text, and category-specific `metadata`
with a strict schema. Priority has no package-level default other than zero and
cannot be changed by the entry function. Aliases, when supported, are declared
as a sorted unique list. Canonical IDs are resolved first. An alias normalizing
to its own canonical ID is redundant and rejected; an alias normalizing to any
other declared canonical ID is invalid regardless of priority. Only alias-
versus-alias conflicts use the normal priority algorithm.

Category metadata supports discovery, help, generated documentation, and
side-effect-free inspection without loading code. After loading, the host
requires the extension's actual category/name registrations to match its
manifest exactly. An undeclared registration, missing registration, or category
mismatch rejects the complete package. Priority exists only in the manifest-
bound registrar; no plugin-supplied priority exists to compare.

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

Missing default directories are ignored. An unreadable existing default or
environment directory is fatal because silently omitting installed code would
change winners. An invalid or empty `AIPERF_PLUGIN_PATH` element is fatal. Every
invalid explicit path is fatal. `--no-auto-plugins` disables only optional
system/user sources 2 through 4; the authenticated distribution-required source
1 is always active for normal product commands. A manifest-only inspection may
omit it only with a separate explicit diagnostic flag that cannot execute a run.
Environment and explicit sources remain active. Internal re-execution never consumes
mutable CLI or environment discovery inputs; it receives the exact parent lock
and a dedicated bootstrap list of locked generation identities.

`--plugin-lock <lock-bundle>/plugin.lock` selects hermetic composition. The lock
file and sibling content-addressed `store/` form one no-follow acquired bundle;
the canonical lock itself contains no absolute paths. In this mode ordinary
system/user/environment discovery and `--plugin-path`/`--plugin-manifest` are
rejected as conflicting inputs, while the executable's authenticated
distribution baseline is still verified. Every manifest, final artifact,
package-build record, baseline requirement, winning/shadowed/ambiguous/
quarantined status, and authority decision MUST appear exactly in the lock;
missing/unreadable inputs appear as typed absence/failure receipts rather than
imaginary artifact bytes. Missing or extra generations, a recomputed quarantine reason that differs, or
any digest/descriptor/host-universe difference fails the command; nothing is
silently rediscovered, repaired, promoted, or omitted. Re-exec children inherit
the parent's acquired staged authority rather than reopening the lock bundle.

Directory entries and normalized manifest identities are sorted before
resolution, so filesystem enumeration order has no semantic effect. Repeated
identical package manifests are deduplicated by canonical manifest and complete
artifact-closure digest. Authority is assigned after deduplication: if any
occurrence is explicit, that canonical package is required for the invocation.
Different manifests claiming the same normalized package name and canonical
version form an identity conflict. Every optional claimant is quarantined as
one conflict group before priority resolution. If any claimant is distribution-
required or explicit, composition fails. No bytes from that group are loaded
and no claimant can win or be promoted.

Only files named `plugins.yaml` or ending in `.plugins.yaml` are candidates.
Discovery is non-recursive. Manifest path order is retained solely for
diagnostics; it never decides a winner. Package identity is the normalized
package name plus canonical package version. Two manifests with that identity
are identical only when their canonical decoded forms and complete artifact-
closure digests match. Multiple versions of one normalized package name are
distinct candidates and then obey ordinary key priority/tie rules.

Priority is resolved independently for each normalized `(category, name)`:

- the highest compatible priority wins;
- an equal-priority tie is ambiguous and makes that component unavailable;
- external plugins must declare a higher priority to replace a first-party
  implementation;
- the frozen catalog records the winner and every shadowed or unavailable
  candidate.

The resolution algorithm is fixed:

1. Acquire discovery inputs once and strictly decode every candidate manifest.
2. Deduplicate candidates, assign optional/distribution/explicit authority, and
   acquire complete immutable artifact closures without invoking a loader.
3. Statically inspect paths, artifact kinds, digests, embedded build records,
   dependency graphs, loader identities, allocator imports, panic policy,
   exports, and search policy. Optional candidates failing this side-effect-free
   phase are quarantined; required candidates fail composition.
4. Resolve canonical IDs first. Reject alias-to-canonical collisions, then group
   remaining aliases independently by normalized `(category, alias)`.
5. Find the maximum signed priority in each group. A unique maximum wins and
   lower candidates are shadowed; an equal maximum is ambiguous with no winner.
6. Establish the immutable intended catalog and load set: every required package
   and package owning a winner. A fully shadowed optional package is not called;
   a fully shadowed required package is still activated and validated because
   package authority is independent of key selection.
7. Begin native activation. From the first platform-loader call onward, any
   dependency mapping, initializer, entry-symbol, descriptor, registration, or
   abnormal termination while dropping activation-stage values poisons
   composition. No candidate is quarantined, priority
   is not recomputed, and no lower candidate is promoted after this point.
8. Validate registrations transactionally, freeze winners, ambiguities, shadows,
   quarantine causes, actual descriptors, and the complete load set into one
   catalog, then derive the canonical lock.

Step 7's no-promotion rule prevents a broken override from silently changing a
run to a different implementation. A later fresh process may obtain a different
catalog only when a discovery-policy input, installed generation, environment
input, explicit input, host identity, or distribution inventory changes.

Registration is manifest-bound. When an extension registers a factory, the
registry obtains its priority and package provenance from the active package
context rather than trusting plugin-supplied ad hoc values. The package is
staged transactionally; an error commits none of its entries. Priority
was already fixed before activation; registration verifies the package's actual
entries against that precomputed winner/shadow/ambiguity map and then commits
the already-selected entries without rerunning or changing priority resolution.

Required first-party packages and required component keys are separate fields in
the authenticated AIPerf distribution inventory. A required package need not
win a key; a required key must have one unambiguous winner. A plugin manifest
cannot grant either authority. An explicit manifest makes the deduplicated
package required for that invocation. The inventory contains canonical package
manifests, complete distribution-controlled non-system artifact closures, the
host ABI universe ID, every package artifact-build ID, component keys, and its
authentication root, but no local absolute path. The installer verifies
inventory authenticity before atomic publication. A partial or unauthenticated
required generation fails composition.

## Composition and lifecycle

Application composition follows a strict sequence. Operating-system loading of
the executable and its distribution-baseline dependencies necessarily precedes
AIPerf code and is the sole exception to the composition-before-host-effects
ordering; AIPerf verifies that preexisting state before discovery:

1. Enumerate every already-loaded non-system module, seed the process-global
   loader-identity-to-digest map with the executable and authenticated
   distribution-baseline modules, and verify every baseline requirement,
   including the allocator. Any other preloaded non-system module—including one
   introduced by `LD_PRELOAD`, `DYLD_INSERT_LIBRARIES`, AppInit, or an equivalent
   injection seam—fails startup before discovery unless its exact identity and
   digest are authenticated in the target's distribution/system policy.
2. Acquire discovery inputs and strictly decode manifests.
3. Acquire and statically validate every complete immutable artifact closure.
4. Assign authority, resolve canonical IDs and aliases, and fix priorities.
5. Commit the immutable intended catalog and load set; resolution ends here.
6. Stage the exact verified closure and begin native activation using restricted
   eager loader rules.
7. Resolve and invoke `aiperf_plugin_entry_v1` on the exact library handle.
8. Apply manifest-bound transactional registration and verify every descriptor.
9. Freeze `AIPerfRegistry`, gRPC endpoint bindings, exporter factories and
   capture vocabulary, direct-execution bindings, and complete provenance into
   one universe. Run-specific exporter capture plans are not frozen here.
10. Derive and commit the canonical plugin lock.
11. Only then construct host runtimes, workers, artifacts, network clients,
    dashboard/control hooks, cells, or benchmark effects.

These states are represented by distinct Rust types:

```text
DiscoveredCatalog -> AcquiredCatalog -> StaticallyValidatedCatalog
                  -> IntendedCatalog -> ActivatingCatalog
                  -> FrozenPluginUniverse -> Application
Application + normalized run config -> ValidatedRunPlan -> PreparedRun
                                                    -> ExecutedRun
```

Mutation methods exist only on `RegistryBuilder` and the package-scoped
`PluginRegistrar`. `FrozenAIPerfRegistry` exposes lookup and catalog methods but
no registration or mutable category accessors. Freezing consumes the builder;
there is no thaw operation. `Application` remains the process-universe wrapper.
`ValidatedRunPlan` owns factory-produced validated
configuration and the combined exporter capture plan for exactly one run. The
public parent creates and hashes a validation-only `ValidatedRunPlan` after
composition and Config-v2 normalization but before artifact logging, dashboard
creation, reset-KV/server-profiler control-hook networking, or child launch. The
child verifies the universe lock, reads the benchmark request, independently
recreates the same plan, and compares its digest before constructing its
`PreparedRun`. A mismatch fails before child effects. Each plan is immutable
before any effect in its process.

The plan digest never serializes or introspects plugin-owned opaque validated
state. Every validation method returns that opaque native value together with a
host-owned `FactoryValidationReceiptV1` containing the selected category and
canonical factory ID, frozen descriptor digest, BLAKE3 digest of the host's
canonical normalized authored config bytes, plugin-computed semantic-config
digest after defaults/normalization, sorted host-resource requirements, and—for
exporters—the exact canonical capture requirement. The plan digest covers the
canonical normalized run DTO plus the sorted receipts. The child reruns factory
validation and requires byte-identical receipts; it constructs a fresh local
opaque value and never transfers the parent's value. A conforming factory MUST
produce the same semantic digest and behavior for the same common universe,
descriptor, and canonical input. Repeatability and default-value mutation
fixtures enforce that rule; receipts are trusted declarations, not a sandbox
against malicious native code.

Production composition is process-global. The first platform load creates an
unsealed process-resident `ActivatingLibrarySet`. Successful registration and
freeze seal it as `LoadedLibrarySet` with the subsequently derived plugin-lock
digest. Failure retains every handle the loader actually returned in a
`PoisonedLibrarySet` with the
original activation error and no fictitious completed lock. A second
composition request in the same process may reuse a successfully sealed set
only when the requested lock digest is identical; a different digest is an
error. A poisoned set rejects every reuse with its original failure. Library handles are
intentionally retained until operating-system process teardown and are never
passed to `Library::close` or dropped during normal Rust destruction. Tests that
exercise loading, failure, or differing catalogs run in subprocesses. This rule
eliminates dependence on registry-clone or factory-product drop ordering.

Before any pointer, descriptor, or object from a plugin becomes reachable, its
handle is committed to the process-resident set. If failure occurs after the
first platform-loader operation, the process-global loader is poisoned. Every
successfully obtained handle remains resident, every later composition returns
the original failure, and the process cannot execute a benchmark. A platform
loader may execute an initializer and then fail/unmap that module without ever
returning a retainable handle (for example a rejected Windows
`DLL_PROCESS_ATTACH`); AIPerf cannot prevent that OS-owned unmap, but no symbol
was resolved and no pointer or value escaped. Loader failure can follow
dependency mapping or native initializers, so the poison boundary is loader
entry, not entry-symbol invocation.

Platform dependency loading is constrained as follows:

- Linux uses `libloading::os::unix::Library::open` with `RTLD_NOW | RTLD_LOCAL`,
  never the lazy cross-platform default. Every staged ELF object uses an
  authored `$ORIGIN` runpath; the loader does not mutate `LD_LIBRARY_PATH`.
  A deliberately retained handle is mandatory on every Unix platform;
  `RTLD_NODELETE`, where supported, is an additional defense and not a
  substitute for retained ownership.
- macOS uses eager local loading and `@loader_path`-relative install names for
  every staged Mach-O dependency; it does not mutate `DYLD_LIBRARY_PATH`. The
  final rewritten and signed artifact is the artifact hashed in the closure.
- Windows loads the fully qualified staged plugin through
  `libloading::os::windows::Library::load_with_flags` using
  `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` plus only the target's approved system
  flags. It never uses `AddDllDirectory`/`LOAD_LIBRARY_SEARCH_USER_DIRS` as
  package isolation. A process-global case-insensitive module-basename-to-digest
  map rejects conflicting non-system DLLs before native activation, and every
  successful module is pinned with `Library::pin`.

Before loading, staging constructs one process-wide canonical object map keyed
by `(loader identity, digest)` whose values carry origin
`Executable`, `Baseline`, or `CanonicalStage`. Every byte-identical package
claimant, regardless of package or platform, resolves to the same host-owned
`CanonicalStage` file and absolute load path; the object is mapped once and its
retained handle satisfies every claimant edge. A package dependency may reuse a
`Baseline` object only through an explicit typed baseline requirement, never
merely because its bytes match. Package-private logical paths remain provenance
only. The map is seeded solely with the verified executable/baseline modules,
adds canonical staged objects before native activation, and rejects an identity
paired with any other digest or an unauthenticated preloaded origin. This
coalescing rule is identical for ELF, Mach-O, and case-folded PE identities and
eliminates loader-dependent accidental reuse of a different package's path.

The SDK statically validates these policies and every dependency edge. Runtime
validation records actual loaded module identities, paths, and digests. A
plugin-private module's path and digest MUST equal its `CanonicalStage` object.
A shared
distribution-baseline module is instead satisfied only by the exact preverified
`(loader identity, digest)` already present in the seeded map; its mapped path is
not expected to equal a nonexistent per-plugin staged path. Any already-loaded
non-system identity not admitted by the baseline or an identical staged closure
is a collision. A plugin MUST NOT rely on current directory, ambient path
variables, process-global symbol lookup, or an already-loaded different module.

No execution path may construct a fresh built-in registry. Root help and
manifest-only inspection need not load libraries. Capability validation and
execution use the exact frozen `PluginUniverse`.

The command behavior is explicit:

- root `--help` and shell-completion generation do not discover or load plugins;
- `aiperf plugins list` discovers and decodes manifests without executing code;
- `aiperf plugins validate` performs closure, build-identity, dependency,
  allocator, panic, entry-symbol, and registration checks only for the active
  intended load set and therefore executes trusted initialization for winners
  and required packages, never fully shadowed optional packages. Validating one
  otherwise-shadowed package requires explicit `--plugin-manifest`, which makes
  it required for that diagnostic invocation;
- `aiperf plugins lock --output <new-directory>` performs the same complete
  composition/activation as `validate`, freezes actual descriptors and status,
  then writes `plugin.lock` plus the complete `LockedCatalogBundle` store to a
  private sibling temporary directory and atomically publishes the requested
  directory. The output path MUST be absent, outside every active discovery
  root, and on one filesystem; the command never overwrites or merges a bundle.
  It executes trusted plugin code, reports that fact before activation, and is
  the sole supported producer for hermetic lock bundles;
- `aiperf config`, profile validation, execution, and native evaluation compose
  before opening artifacts or starting dashboards, control-hook networking,
  Tokio/Velo runtimes, dataset acquisition, or any registered component;
- commands unrelated to registered runtime components do not load plugins
  unless they request the capability catalog.

The public parent composes before artifact logging, dashboard creation, control-
hook networking, or child launch. Ordinary `--execute` gains a dedicated
inherited private bootstrap authority carrying the canonical lock DTO, expected
validated-run-plan digest, and
handles to the parent's complete `LockedCatalogBundle`; benchmark
stdin remains unchanged. On Unix these are no-follow directory/file descriptors
with explicit inheritance. On Windows these are explicitly inheritable handles
named in the child process attribute list; ambient handle inheritance is
disabled. The channel is distinct from the existing cell-security descriptor.
The child does not rediscover, reopen an explicit outside-root path, or race an
installed generation; it rehashes all catalog inputs and loads only load-set
objects reachable through the inherited authority, then proves the full lock before reading the
benchmark request.

Same-host cells receive the expected lock, the canonical normalized full run
DTO, deterministic cell partition identity/rules, and expected cell-specific
validated-plan digest through their private bootstrap pipe. Cross-host
Kubernetes/SLURM launch material supplies the same values in a fixed-`0600`,
no-follow bootstrap file rather than argv or environment, plus the complete
locked-catalog inventory. The controller derives and hashes every cell slice
before launch. A cell composes and derives its slice immediately after
reading bootstrap material and before creating Tokio/Velo runtimes, dialing,
fetching datasets, opening artifacts, or joining barriers. `CellRegister` gains
the lock and validated-plan digests; the signed registration transcript binds
both, and controller
registration verifies it transactionally before routes, artifact authorization,
or barrier state commit. Remote hosts preinstall exact artifacts; automatic code
transfer remains absent.

Velo `RegisterReply` no longer provides the first authoritative cell run
configuration. It repeats the prebootstrapped slice and plan digests for
authenticated agreement; a difference is a registration failure. Plugin
factories validate the local slice and reproduce byte-identical receipts before
the Velo runtime exists. This is an explicit migration from the current
register-then-compose cell order.

This requires explicit product-schema migrations, not an assumption that the
present launch DTOs already carry plugin state. `CellLaunchContext` gains the
expected canonical lock, normalized run DTO, cell partition, validated-plan
digest, and complete locked-catalog inventory. The native
Kubernetes envelope, image-capabilities document, operator-owned JobSet pod
specification, controller/cell bootstrap schema, and results provenance all bind
the distribution generation, host ABI universe, plugin lock, and immutable
artifact inventory. Image-capability validation proves those generations are
present before cluster effects. Native SLURM `run` and `generate` materialize
the same fields for every sibling rank before `srun` begins; peer startup order
is not used as distribution or lock authority. Older envelopes/bootstrap
schemas that cannot express these fields fail closed for plugin-enabled runs.

`LockedCatalogBundle` is distinct from the intended load set and from
distribution-required package authority. Its manifest record is presence-
tagged: readable inputs contain exact raw bytes/digest; successfully decoded
inputs additionally contain canonical bytes/digest; malformed inputs have no
canonical form; and unreadable directory entries have neither raw nor canonical
bytes. It contains every successfully acquired
closure object needed to recompute its digest, including fully shadowed optional
packages; stable typed acquisition/static-validation receipts for quarantined
inputs; discovery-policy and authority inputs; the canonical status table; and
the canonical object map. It never executes a quarantined or fully shadowed
optional artifact. Parent, re-exec child, same-host cell, and cross-host image
all receive or preinstall this complete immutable bundle and independently
recompute the full catalog lock. A required-generation-only or load-set-only
inventory is non-conforming because it cannot reproduce shadowed, ambiguous, or
quarantined lock entries. A failure receipt binds the canonical discovery-
source identity and normalized relative entry identity, explicit raw/canonical
presence tags and their digests when present, attempted logical object identity,
stable error code, and available acquired metadata; it never fabricates a raw
digest or asks a child to recheck a mutable missing pathname.

The canonical lock binds the full frozen catalog, not only executable winners.
It contains lock schema and normalization versions; host ABI universe ID;
every manifest's raw/canonical presence tags and corresponding digests when
present; canonical package name/version/authority when decoded;
complete distribution-controlled non-system artifact-closure digests and every
plugin artifact-build ID; verified baseline module identities/digests; package
authority/load status;
per-entry status `winning|shadowed|ambiguous|quarantined`; the complete canonical
sorted failure-receipt set; every actual registered descriptor digest; canonical and alias
winner maps with priorities; required package identities; required component
keys; and target system-library allowlist version. Malformed readable manifests
use their raw-byte digest and stable failure code; unreadable entries use their
canonical discovery-source/relative-entry identity, absent-raw marker, and
stable failure code. It excludes absolute paths and free-form
diagnostic text. The digest is BLAKE3 over canonical length-delimited bytes.
Every process rebuilds the full lock and reports the first structured difference
before execution.

Failure selection never depends on checker iteration order. Every discovered
failure contributes one receipt sorted lexicographically by
`(phase_ordinal, discovery_source_id, relative_entry_id, logical_object_id,
error_code, evidence_digest)`; optional values use explicit absence tags. Phase
ordinals follow the numbered resolution algorithm. `discovery_source_id` is the
canonical length-delimited `(source_kind_ordinal, authored_index)` where kinds
are distribution, platform-system, platform-user, environment, explicit-
directory, explicit-manifest, or hermetic-bundle; each ordered root/argument has
its zero-based index. It contains no local path. Directory-discovered manifest
basenames MUST be lowercase ASCII and match either `plugins.yaml` or
`[a-z0-9][a-z0-9_.-]*.plugins.yaml`; this exact basename is
`relative_entry_id`. An explicit-manifest entry uses its authored argument index
as the relative ID, so an unreadable platform pathname need not be encoded.
`logical_object_id` is the manifest-declared normalized artifact identity when
available. The evidence digest covers the presence-tagged evidence record. The
lock retains all receipts; it never chooses one "first" quarantine reason.

Automatic executable transfer is intentionally absent. Kubernetes, SLURM, and
other cross-host distributions MUST package the same plugin artifacts on every
participating host or image. The controller never accepts a merely compatible
but byte-different plugin lock.

### Panic and ownership behavior

The linkage rule is absolute: no host/plugin boundary uses `catch_unwind`, and
every shipped artifact uses `panic=abort`. A panic, explicit abort, loader
termination, undefined behavior, or memory corruption may terminate the process;
the system makes no isolation claim. Recoverable conditions use typed errors.

No borrowed value received from a plugin may be retained beyond the lifetime
declared by its native trait method. Any `'static` descriptor from a plugin is
valid because the defining library is process-resident. Futures and callbacks
created by a plugin retain native ownership and finish or are dropped before
process teardown. A plugin MUST NOT spawn a detached task or thread whose code
can run after AIPerf has declared application shutdown complete.

## Runtime category behavior

Only endpoint, transport, and exporter are dynamically loadable in API
generation 1. Dataset loaders, samplers, workloads, actuators, native-graph
factories, clocks, and observers remain host-owned or statically composed.
Transport-specific direct/co-simulation execution is a transport sub-capability,
not a fourth category. Adding a fourth dynamic category requires a new reviewed
API generation or an explicitly backward-compatible source API addition plus
new manifest schema, conformance, lifecycle, and performance coverage.

The manifest describes discoverable capabilities, but executable descriptors
returned by native factories remain authoritative after successful
registration. Composition rejects a discrepancy rather than merging metadata
from two sources. Configuration never selects a Rust type name, symbol name,
package path, or library path; it selects a normalized category ID from the
frozen catalog.

### Endpoints

Endpoint plugins register one atomic endpoint capability containing the native
`EndpointFactory` contract and optional transport-specific binding factories.
Each worker
prepares its existing worker-local `PreparedEndpoint` once. Formatting and
streaming response parsing continue to use native AIPerf values and byte
buffers. No FFI container, serialization, or additional adapter call is added.

Endpoint canonical IDs and aliases remain open strings under the canonical-first
rules. Alias shadowing is explicitly partial: a factory can win its canonical ID
while another higher-priority canonical winner owns a shared alias. The authored
descriptor records all claimed aliases; the frozen effective descriptor and
lookup table retain only aliases that entry actually wins. Companion transport
bindings follow the canonical factory, never the alias winner independently.
Generic endpoint policy remains host-owned;
endpoint-specific configuration validation and request/response formatting
remain plugin-owned. Streaming parsers MUST continue preserving incomplete byte
sequences across chunks and MUST emit observations through the existing host
observer seam.

The endpoint migration preserves three distinct shapes and MUST NOT conflate
them. Authored `cfg.endpoint` is the default profile object
`{type: <endpoint-registry-id>, ...fields}` and has no authored profile `id`.
Authored `cfg.endpoint_profiles` is a map whose key is the additional run-local
profile ID and whose value has the same `{type, ...fields}` shape. Protocol-v2
projection alone produces the ordered `profiles` records and injects `id:
"default"` or the authored map key into each record. `type` names the endpoint
factory in all three shapes. Its existing transparent string `EndpointType`
remains the source spelling and normalizes to the open registry ID; the seams
removed are the monolithic typed `Endpoint` field schema and downstream hard-
coded capability matches, not a closed ID enum.

The compatibility decoder separates documented host-owned connection/policy
fields from the remaining raw factory object before strict factory validation.
The current closed typed top-level `Endpoint` is not carried into protocol v2 as
an authoritative second copy. Hard-coded endpoint-name matches in CLI/config
validation become descriptor capability checks; unknown registered endpoint IDs
are not rejected by a closed enum or string list.

The optional `GrpcEndpointBindingFactory` is owned and registered by the same
endpoint package. gRPC unary, server-streaming, and bidirectional codecs resolve
from the exact winning endpoint capability. `GrpcBindingRegistry::builtin()` and
every other fresh binding registry are removed from production validation and
execution; the frozen universe injects one binding table into gRPC execution.
An endpoint override cannot accidentally retain the shadowed implementation's
static gRPC codec.

### Transports

Transport plugins register `TransportFactory` and exactly one declared execution
shape: `RequestTransportExecution` for ordinary scheduled/graph dispatch or
`DirectTransportExecution` for transport-owned direct/co-simulation driving.
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

The canonical internal representation is exactly `{ id, config }`, where `id`
is required after normalization and `config` defaults to an empty object. An
absent authored `transport` retains the current default and normalizes to
`{id: http, config: {}}`; absence is not an unknown third union variant. Config
v2 uses a strict
compatibility union: it accepts either legacy `{type, ...flat_config}` or open
`{id, config}`, never both and never a mixture of `type` with `id`/`config`.
Legacy `type` becomes normalized `id` and all remaining fields become `config`.
The generated schema uses `oneOf`; normalization occurs before protocol-v2
validation; serialization emits the open form. Existing legacy input behavior
is retained, while emitted Config v2 intentionally adopts the open form. The
compatibility decoder remains through the next major Config schema, where its
removal requires a separate migration record.

The factory receives the raw JSON object once during strict startup validation
and returns opaque native validated configuration plus a
`FactoryValidationReceiptV1`. The host passes that value only to the exact
factory instance that created it. No host switch on a closed
transport enum and no `transport_typed` protocol copy may remain in CLI YAML,
control hooks, validation, protocol projection, preparation, execution binding,
capability reporting, or cellular projection.

The transport plugin does not own phase scheduling, request admission, worker
partitioning, metrics, cancellation policy, or clocks. It owns only the
transport-specific execution objects represented by its declared binding.
`DirectTransportExecution` receives narrow host-owned clock, graph, metrics,
artifact, and cancellation service traits; it never receives `RunContext` or
runtime-private prepared operations. `dynosim_offline` and `dynosim_online`
implement this binding, and all `dynosim_or_unsupported!`/closed-enum branches
are removed. A transport that needs a new host capability must add an explicit
reviewed capability contract; it cannot reach into runtime-private state.

### Exporters

Exporter plugins register `ExporterFactory` and supported capture vocabulary
during process composition. Capture requirements are run-specific: after the
frozen universe exists and the run configuration has been read and normalized,
each selected factory strictly validates its opaque config and returns an
opaque value plus `FactoryValidationReceiptV1` whose capture field is an
`ExporterCaptureRequirementsV1` value from this closed host-owned vocabulary:
`FinalReport`, `ExactRecordsV1`, and
`FoldedProjectionV1(GenAiClientHistogramsV1)`. A requirement is a sorted set of
those values, so a factory may request a defined union but cannot invent a
projection name or schema. `ExactRecordsV1` is the existing full canonical
native captured-record DTO sequence, including warmup and profiling records
with their `benchmark_phase`; each exporter retains its existing phase-filter
policy, and only the folded histogram projection is profiling-only. The exact
record sequence has an explicit order per execution family:
all scheduled workload families, including user-centric and their sharded or
cellular forms, sort native captured records by ascending non-absent
`request_index` and then UUID; the separate `outputs.json` projection may retain
its existing `(session_num, turn_index)` order but does not define
`ExactRecordsV1`. Non-cellular graph records sort by `(start_ns, UUID)` and then
receive dense `request_index` values; cellular graph
records concatenate ascending `cell_id`, sort within each cell by that cell's
local `request_index` and UUID, and then receive dense controller-global
`request_index` values. The latter intentionally remains deterministic per
topology rather than claiming single-cell byte order. A missing required key or
duplicate full ordering key is a typed capture failure. This preserves existing
family-specific order; it does not redefine all records as admission ordered.
The host combines requirements into an immutable
`ValidatedRunPlan` and only then installs its existing per-worker capture/fold
path and creates runtime effects. Capture plans are not part of the process-
global plugin lock. No plugin callback, allocation, dispatch layer, or plugin-
specific accumulator is added per record solely for export. After report
commit, a prepared exporter receives the finalized native report, generic host-
owned captured projection, and capability-limited `ArtifactAccess` that permits
scoped list/read of already committed run artifacts plus create/write under
approved relative destinations. It exposes no raw artifact-directory path or
unchecked join.

`GenAiClientHistogramsV1` is the sole folded projection in API generation 1.
Its versioned native Rust schema is a sorted map from
`(metric, capture_dimensions)` to `ExplicitHistogramV1`, where `metric` is exactly one
of `gen_ai.client.operation.duration`,
`gen_ai.client.operation.time_to_first_chunk`,
`gen_ai.client.operation.time_per_output_chunk`, or
`gen_ai.client.token.usage`. Capture dimensions contain only host-observed per-
record distinctions: duration keys carry optional normalized `error_type`
exactly where the existing record projection emits it, while token-usage keys
carry `token_type` equal to `input` or `output` and never carry `error_type`.
Operation name, provider name, and optional request model are deliberately not
capture dimensions because they come from each selected exporter's opaque
configuration and multiple OTLP instances may differ while sharing one fold.
Each histogram contains immutable finite ascending `f64` bounds,
`bounds.len() + 1` `u64` bucket counts, checked `u64` count, finite `f64` sum,
and optional finite min/max. The SDK publishes the exact metric-source aliases,
unit conversions, bounds arrays, success/error inclusion rules, and attribute
normalization as constants in `aiperf-core`; factories cannot supply them.
Observations use first-upper-bound `value <= bound` semantics. Worker, shard,
and cell merge requires identical keys and bit-identical bounds, adds bucket
counts and counts with checked overflow, adds sums using the host's fixed
deterministic merge order, and takes min/max. A key/bounds mismatch, overflow,
or non-finite result is a typed execution failure before export.

At export, each prepared OTLP instance independently decorates every projected
histogram data point with its normalized `gen_ai.operation.name`,
`gen_ai.provider.name`, and optional `gen_ai.request.model` from that instance's
validated configuration, maps `error_type` to `error.type`, and maps
`token_type` to `gen_ai.token.type`. Its semantic-config digest binds those
values. Decoration performs no per-record work and does not require a distinct
capture fold per exporter instance.

Run-plan validation computes whether the selected workload, retention mode, and
cellular topology can satisfy the union. `FinalReport` is always available.
Selecting an exporter that declares `ExactRecordsV1` is explicit user consent
to exact retention: the validated plan sets host-owned retention reason
`RequiredByExporter(<exporter-id>)` and feeds it into the same planning decision
as per-record artifacts, overriding the default exact-fold/no-record
optimization. No separate environment-only control is required. An explicit
sketch policy or execution mode that cannot produce the exact canonical
sequence conflicts and fails before runtime construction with the exporter ID
and remediation; selection never silently disables sketch or substitutes a
fold. `GenAiClientHistogramsV1` is folded once
per completed profiling record and is supported in retain, exact-fold, sketch,
sharded, and cellular modes through worker-local accumulation and deterministic
boundary merge. Warmup/excluded records follow the existing metrics-plane scope
and do not enter it. Any selected exporter requirement that cannot be satisfied
fails run validation with exporter ID, requirement, and conflicting mode.

Cellular `ExactRecordsV1` requires an explicit protocol migration. The current
`RecordsShardPartition<Vec<RecordIngest>>` remains the compact metrics path and
cannot satisfy exact capture because it omits UUID, correlation ID, output/
reasoning text, and raw exchange. When and only when the validated union requests
`ExactRecordsV1`, each cell instead emits versioned
`ExactRecordsPartitionV1` chunks containing its `cell_id`, monotonically dense
chunk sequence, declared record count and byte length, BLAKE3 payload digest,
and complete public `ExactRecordV1` DTOs (the full captured-record projection,
including optional output/raw fields only when the host capture policy permits
them). Chunks use the existing authenticated cell route, configured finite byte
and record bounds, and bounded backpressure; the controller rejects gaps,
duplicates, digest/length/count mismatches, unexpected cells, or resource-limit
excess before exporter preparation.

The controller reassembles each cell sequence, applies the family-specific
canonical ordering above, and verifies exactly one record per declared identity
before exposing the projection. Same-host and cross-host cells use the same wire
DTO and merge rules. Runs not requesting exact records retain the existing
`RecordIngest`/folded-store partitions and pay no exact-record transfer cost.
This host-owned post-run transfer adds no plugin callback or request/token-path
operation, but its bandwidth/memory and parity are separately benchmarked.

Every cellular terminal payload also carries a versioned host-owned
`CellCaptureBundleV1` independent of whether the metrics payload is current
`Records` or `Store`. It binds the cell's validated-plan digest and contains
exactly one presence-tagged result for every folded projection in that plan;
generation 1 therefore carries an empty or populated
`GenAiClientHistogramsV1` DTO with its projection ID, schema version, bounds,
keys, counts, sums, extrema, and payload digest. A cell cannot omit an expected
empty projection or add an unrequested one. The controller verifies the plan
digest, cell identity, projection set, schema, canonical bytes, and digest before
performing the checked deterministic worker/cell merge defined above. Missing,
duplicate, injected, malformed, or plan-mismatched projection results fail the
run before exporter preparation. This replaces the current local-only transient
OTLP report side channel; neither the `Records` nor folded `Store` metrics mode
may silently drop generic capture state.

`NativeReport` MUST NOT contain an OTLP-specific implementation type. The current
`OtelRecordAccumulator` and `NativeReport::otel_per_record` side channel become
the transport-neutral `GenAiClientHistogramsV1` capture result selected by the
OTLP factory's requirement and supplied beside, not embedded in, the committed
report. Exact, sketch/folded, sharded, and cellular merge semantics are defined
by the host projection so OTLP remains a separate plugin without leaving its
implementation in the host.

Fixed exporter enablement fields become an ordered open selection:

```yaml
exporters:
  - id: genai_perf_v1
    config:
      json: true
      csv: true
  - id: otel
    config:
      endpoint: http://collector:4318
```

Config v2 accepts the legacy `cfg.export` fixed object, the new `exporters` list,
or neither, but never both. Legacy enabled fields project to instances in the
existing `ExporterRegistry` order (ascending existing order key, then normalized
exporter ID); disabled fields emit nothing. Generated schema expresses this as
three mutually exclusive branches rather than a two-branch `oneOf` that would
reject the valid neither case. Normalization precedes protocol projection, and
serialization emits the open list when exporters are selected and omits it when
none are selected. Existing CLI flags project through the same legacy
normalizer. Each selected factory strictly decodes only its own `config` object.

First-party exporter identities and legacy projection are frozen as follows.
`file+n` and `uploader+n` mean order keys `n` and `1000+n`, respectively. Each
enabled legacy source produces exactly one exporter instance; JSON/CSV toggles
remain fields of that instance and never expand into separate `json` or `csv`
IDs. Absent/disabled sources produce no instance.

| Canonical ID | Legacy `cfg.export` source | Canonical open `config` | Order key |
|---|---|---|---|
| `genai_perf_v1` | `genai_perf` | Exact normalized `GenaiPerf` object, including `json`/`csv` toggles | `file+0` |
| `server_metrics` | `server_metrics` | Exact normalized object, including `json`/`csv` toggles | `file+1` |
| `timeslice` | No authored fixed field; current planner-derived internal config | Exact normalized current `TimesliceExportConfig` when enabled | `file+2` |
| `accuracy_csv` | No authored fixed field; accuracy-mode planner enablement | Exact normalized current `AccuracyCsvExportConfig` when enabled | `file+3` |
| `server_metrics_parquet` | `parquet` | Exact normalized `ParquetExport` object | `file+4` |
| `console_txt` | `console_txt` | Exact normalized `ConsoleTxt` object | `file+5` |
| `otel` | `otel` | Exact normalized `OtelExport` object | `uploader+0` |
| `mlflow` | `mlflow` | Exact normalized `MlflowExport` object | `uploader+1` |
| `wandb` | `wandb` | Exact normalized `WandbExport` object | `uploader+2` |

The package/artifact may use the descriptive term OTLP, but the canonical
registry/config/report ID remains the existing `otel`. Generation 1 defines no
`json` or `otlp` alias. Using either as an ID fails with a diagnostic naming the
valid canonical ID; legacy `cfg.export.otel` normalizes directly to `otel` with
no rename. Changing, splitting, or aliasing one of these identities requires a
separate compatibility migration.

The canonical exporter representation is an authored ordered list of
`{ id, config }`, with an empty object default for `config`. The host resolves
every authored canonical/alias spelling through the frozen catalog first, then
API generation 1 rejects duplicate resolved `(canonical factory ID, descriptor
digest)` identities in one run. Two aliases of one winner therefore cannot
instantiate it twice. Multi-instance exporters are deferred to a later API
generation with an explicit instance-key contract. Registry priority selects an
implementation; list order does not override package priority.

Exporter execution order is still host-owned. Each exporter descriptor declares
its order band and stable tie-break key. The host sorts enabled
instances deterministically and preserves the rule that local artifact writers
complete before uploaders. A plugin cannot access paths outside the host-created
artifact directory through an unchecked join API.

Exporter failures retain the current best-effort continuation semantics: the
committed report is not invalidated and remaining exporters continue in
deterministic order. Persisting each failure in the run result and terminal
artifact metadata is an explicit diagnostic migration, because the current
registry only logs the error and returns a success count that callers discard.
The new exporter runner returns a structured ordered outcome list; the
coordinator and cellular merge paths commit that list to terminal/result
metadata without converting it into benchmark failure. This remains distinct
from a transport or endpoint runtime operation failure. Changing exporter
failure to fail the benchmark is out of scope and requires a public behavior
change.

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
from hiding local regressions. Failure to demonstrate simultaneous one-sided
non-inferiority within every band blocks migration; absence of a statistically
significant regression is not a pass.

The initial acceptance protocol is normative:

- compare the dynamic distribution against a test-only monolithic baseline built
  from the same source revision, compiler, target, optimized profile, affinity,
  dependency versions, and implementation sources. The baseline preserves
  static registration, fat LTO, and the pre-plugin statically linked mimalloc
  topology; only artifact/linkage topology differs. It is never shipped;
- run on an otherwise idle pinned host with fixed CPU frequency policy; pin the
  in-repo mock server to disjoint isolated cores or a separate pinned host;
- execute exactly 30 retained paired samples after warmup for each representative HTTP
  non-streaming, HTTP streaming, gRPC, multi-worker, and exporter workload that
  applies to the migrated component;
- give each case a frozen successful-request budget sized before comparison so
  the static baseline's median sample lasts at least 30 seconds. A dynamic-only
  product error, crash, timeout, incomplete request budget, or malformed output
  fails the component gate immediately and is never retried away. A product
  error in both members is also a gate failure. Only a blinded harness-owned
  infrastructure classifier fixed in `plugin-parity.yaml` before measurements
  begin may invalidate a pair—for example host reboot, loss of required CPU
  affinity, or mock-server death unrelated to either member. Invalidation
  replaces the whole AB/BA pair, retains both raw attempts and the reason, and
  reruns the replacement in the invalidated pair's same member order so the 30
  retained pairs remain exactly balanced. Replacement is capped at five pairs
  per case; exceeding the cap invalidates
  the complete experiment and requires a fresh run after diagnosis;
- randomize a reproducible balanced AB/BA order within pairs so each artifact
  runs first in exactly 15 pairs, recording the seed and complete order;
- compute one summary per run, use Hyndman-Fan type 7 percentiles, and bootstrap
  the 30 pairs rather than individual requests with at least 100,000
  deterministic resamples;
- derive one-sided 95% simultaneous non-inferiority lower bounds with a paired
  max-degradation bootstrap across every normative metric and case for that
  component; separate per-metric intervals do not establish the aggregate gate;
- define throughput ratio as `dynamic / static` and require the lower endpoint
  of its 95% interval to be at least `0.99`;
- define each latency and CPU ratio as `static / dynamic` and require the lower
  endpoint of its 95% interval to be at least `0.99` for TTFT p50/p90/p99,
  inter-token-latency p50/p90/p99, and CPU time per successful request
  (equivalently, dynamic may not be more than 1% worse);
- define exporter duration ratio as `static / dynamic` and require the lower
  endpoint of its 95% interval to be at least `0.99` for exporter nanoseconds
  per record;
- require no increase in allocation count or allocated bytes per successful
  request in deterministic endpoint, transport-dispatch, response-reduction,
  and exporter-capture microbenchmarks;
- require no new lock acquisition, allocation, serialization, copy, task spawn,
  channel operation, or dynamic-dispatch layer in a code-path inspection of the
  migrated request/token path. The inspection emits named compiler IR/call-
  graph and allocation/lock instrumentation artifacts so the negative claim is
  reproducible.

The allocator provider is not exempt. Existing Rust allocation shims must bind
directly to the shared provider's mimalloc entry points without an additional
Rust wrapper or dispatch table, and its call path, throughput, CPU, allocation
count, and allocated bytes are compared against the monolithic allocator
baseline. Failure blocks the architecture rather than redefining the baseline.

A benchmark inventory entry names exactly one `primary_metric` from
`successful_requests_per_second`, `output_tokens_per_second`,
`cpu_nanoseconds_per_successful_request`, or
`exporter_nanoseconds_per_record`, and names its non-inferiority ratio direction.
For its 30 retained pairs, the harness computes the sample coefficient of
variation as Bessel-corrected sample standard deviation divided by the absolute
arithmetic mean for (a) the 30 static member summaries, (b) the 30 dynamic
member summaries, and (c) the 30 positive paired ratios in the declared
direction. If any of those three values exceeds 2%, the complete attempt is
invalid, not a pass; every sample remains retained.

The experiment identity is BLAKE3 over one canonical record containing: exact
source tree and Cargo.lock digests; rustc/sysroot/target/profile and every
compared artifact digest; benchmark-harness and mock-server artifact digests;
canonical `plugin-parity.yaml` digest; CPU model/stepping/microcode, core and
memory topology, firmware, kernel, allocator/provider, frequency/governor,
affinity/isolation, and mock-server placement identities; and every environment
value admitted by the harness. Omitted fields are forbidden. A change creates a
new identity only through a reviewed experiment-change record that cites the
prior attempts and explains why the changed field invalidates comparison; it
cannot be used merely to replace a valid failure.

At most three complete experiment attempts are permitted for one experiment
identity. Only an invalid attempt may be rerun after a
documented noise diagnosis. The first statistically valid attempt is
authoritative whether it passes or fails and cannot be replaced by a later run.
A product error is an immediate valid gate failure, not an invalid attempt.
Three invalid attempts block migration until the source, environment, or
inventory identity changes through review. The
harness records raw samples, environment identity, and confidence intervals as
CI artifacts. Threshold changes require an explicit design/specification
change; a migration patch cannot loosen them to obtain a green result.

Before the first dynamic migration, the repository adds
`rust/benchmarks/plugin-parity.yaml` as the immutable comparison inventory. It
contains at least HTTP non-streaming at concurrency 1 and 64, HTTP streaming at
concurrency 1 and 64 with 32 deterministic response chunks, gRPC unary and
streaming at concurrency 1 and 64, a four-worker run, OTLP-disabled and OTLP-
enabled capture runs, and an exporter pass over 100,000 deterministic records.
Each entry freezes request budget, minimum valid duration, core assignment,
mock-server placement, response shape, warmup count, estimator, bootstrap seed,
primary metric and ratio direction, measured metrics, and the infrastructure-
invalidation classifier. Each case performs five unmeasured warmup samples followed
by 30 paired samples. Changing any field is a performance-contract change and
retains the prior result for comparison.

The compile-time goal is also tested structurally. Editing one plugin crate MUST
NOT rebuild or relink the `aiperf` host or an unrelated plugin. A minimal host
build MUST NOT include gRPC, WebSocket, Dynosim, Parquet, OTLP, MLflow, or W&B
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

Once the first platform-loader call begins, every failure poisons composition;
loader initializers may already have run even if symbol resolution later fails.
Transactional staging prevents partial registry visibility but cannot undo
native initializer effects. A benchmark never proceeds after activation fails.
Runtime methods return typed failures and cannot mutate the frozen catalog;
panics abort the process under the mandatory panic policy.

Native plugins are trusted code. They can access process memory and host
resources, perform system calls, abort, or violate memory safety through unsafe
Rust. Digests establish identity and reproducibility, not isolation. Loading and
registration are trusted effects: conforming plugins MUST NOT perform benchmark
work or start detached activity from native initializers, entry, or registration.
The SDK rejects plugin-owned constructor/initializer sections where the platform
format permits reliable inspection, but this rule is conformance, not sandboxing.

Reports preserve executable `distribution_id` semantics and add the required
`plugin_lock_digest`; they do not conflate the two. A detailed catalog artifact
records the full lock status and provenance without absolute discovery paths.
The controller constructs this metadata for merged cellular reports; cells do
not silently omit coordinator/plugin identity.

The failure policy is fixed by phase and authority:

| Condition | Auto-discovered optional package | Distribution-required package | Explicit `--plugin-manifest` |
|---|---|---|---|
| Preloaded distribution-baseline provider missing/mismatched | Abort startup before discovery | Abort startup before discovery | Abort startup before discovery |
| Locked-catalog/bootstrap digest, absence receipt, or plan mismatch | Fail before runtime effects | Fail before runtime effects | Fail before runtime effects |
| Missing default discovery directory | Ignore | Fail if it contains the distribution generation | N/A |
| Unreadable existing discovery directory or invalid environment path | Fail discovery policy | Fail discovery policy | N/A |
| Unreadable/invalid manifest before activation | Quarantine and report | Fail composition | Fail command |
| Missing/wrong artifact kind or closure member | Quarantine and report | Fail composition | Fail command |
| Digest, embedded record, host-universe, or package-build mismatch | Quarantine without loading | Fail composition | Fail command |
| Static dependency/search/allocator/panic-policy failure | Quarantine without loading | Fail composition | Fail command |
| Same package/version claimed by differing manifests | Quarantine every optional claimant | Fail composition | Fail command |
| Same non-system loader identity claimed by differing bytes | Quarantine every optional claimant | Fail composition | Fail command |
| Equal-priority key tie | Record ambiguity; fail only if selected | Fail composition only for a required component key | Record ambiguity; fail if selected |
| Any platform-loader/dependency activation failure | Poison process and fail composition | Poison process and fail composition | Poison process and fail command |
| Entry symbol, descriptor, or registration error after activation | Roll back only the registry transaction; retain staged generations and mapped handles; poison process and fail composition | Same | Same |
| Post-load canonical lock mismatch | Retain mapped handles, poison process, fail composition | Same | Same |
| Panic in any boundary code | Process aborts | Process aborts | Process aborts |
| Selected key is quarantined/ambiguous | Fail validation before effects | Fail validation before effects | Fail validation before effects |
| Runtime trait method returns error | Typed operation failure | Typed operation failure | Typed operation failure |
| Post-report exporter returns error | Record and continue remaining exporters | Same | Same |
| Explicit abort, UB, or memory corruption | Process may terminate | Process may terminate | Process may terminate |

"Quarantine" means the library entry symbol is not called when rejection occurs
before loading, none of its factories enters the active registry, and the exact
cause remains queryable. Quarantine is fixed for the process. There is no retry
after files change on disk.

Diagnostics MUST include category/name when applicable, normalized package name
and version, source API version, the mismatching build-record field when known,
and a remediation telling the user to rebuild or install a matching plugin.
Normal diagnostics omit absolute paths and library bytes. Debug diagnostics may
show a redacted/canonical package location but never dump plugin configuration
that may contain credentials.

Priority is not a security boundary. A higher-priority third-party package can
replace first-party code and then runs with full process authority. Write access
to any enabled discovery root is therefore equivalent to code-execution
authority for processes using that root. Every path component, manifest,
generation directory, and artifact is checked through its acquired handle.
System roots require administrative ownership and reject any POSIX mode bit,
POSIX/NFS ACL, or Windows DACL granting mutation to a non-administrative
principal. User roots and their contents must be owned by the invoking user and
reject group/world mode writes, foreign-user/group ACL writes, inherited broad
Windows write/delete/replace grants, and owner changes during acquisition;
platform-defined administrator/SYSTEM recovery rights do not make an otherwise
user-owned root foreign-writable. All checks are handle-relative and reject
links or reparse points. An ACL that cannot be fully interpreted fails closed.

For setuid/setgid, elevated-token, service, or otherwise privileged execution,
user roots, `AIPERF_PLUGIN_PATH`, and ordinary explicit paths are disabled.
Only the distribution generation and administrator-owned system roots are used;
`--allow-unsafe-plugin-root` is rejected rather than providing an override in
privileged mode. Generation 1 does not authenticate third-party authors: local
installation authority is the trust root. The first-party distribution
inventory digest is embedded in the executable build record and verified before
use. Revocation ships a new executable/inventory generation; rollback requires
an explicit older complete distribution and cannot mix generations.

The loader does not download code, resolve network package repositories, or
modify installations. Installers and documentation MUST state these authority,
signature, rollback, and revocation rules explicitly.

## Migration

0. Complete the native-boundary feasibility gate: four-target `dylib`/`cdylib`
   probe, selected `cdylib` export/import maps, one shared allocator provider,
   `panic=abort`, distribution-controlled non-system closure inspection, and every proposed owned container
   crossing. Public lab commit `f5af252` proves the exact-build mechanical
   ownership, abort, residency, and shared-provider import topology on four
   targets, but does not prove Rust ABI soundness, production closure enforcement,
   or the zero-loss performance gate. No production migration begins until the
   complete production conformance and performance gates pass.
1. Record clean-build, incremental-build, link-time, binary-size, runtime, CPU,
   and allocation baselines and freeze `plugin-parity.yaml`.
2. Extract the ABI-facing API/core crates, narrow host contexts, exact ownership
   table, allocator provider, exporter capture vocabulary, endpoint companion
   bindings, and request/direct transport bindings while implementations remain
   static.
3. Implement content-addressed packaging, authenticated distribution inventory,
   manifest parsing, discovery, priority, complete-closure validation, host-
   universe/package-build locking, eager platform loading, ordinary re-exec bootstrap, cell
   attestation, report provenance, and separately built conformance plugins.
4. Migrate basic post-report exporters, Parquet, MLflow, and W&B independently;
   migrate OTLP only after its implementation-specific accumulator is replaced
   by the host-owned generic capture/fold plan and all execution modes pass.
5. Migrate endpoint families together with their optional gRPC binding factories
   while preserving prepared worker-local behavior and removing every separate
   built-in gRPC binding registry.
6. Migrate transports last: HTTP, gRPC, WebSocket, dry-run, then the two Dynosim
   IDs through `DirectTransportExecution`. Remove every closed enum/switch and
   hidden direct-execution path for a migrated production ID.
7. Remove static fallbacks only after catalog, artifact, packaging, behavior,
   report, cellular, and performance parity pass for every first-party plugin.

Each numbered migration step is independently shippable and leaves one
authoritative composition path. During steps 2 through 6, a component is either
declared static or dynamic in the distribution inventory; it is never
simultaneously registered through hidden static and dynamic paths. A temporary
test-only comparison binary may contain both under different IDs, but production
IDs remain unique.

The exporter migration order is `basic`, `parquet`, `mlflow`, `wandb`, then
`otlp`; telemetry exporters are separate artifacts and do not form one
"telemetry" dependency bundle. Endpoint packaging follows measured dependency
and coupling boundaries rather than forcing one library per dialect. Transport
migration is `http`, `grpc`, `websocket`, `dry_run`, `dynosim_offline`, then
`dynosim_online`; HTTP is the proof that the common online hot path satisfies the
performance contract.

Feature ownership is fixed as follows. The old `grpc` feature splits into the
gRPC transport package plus endpoint-owned KServe/Riva binding packages; the
minimal host owns only transport-neutral traits. The old `parquet` feature
splits into a Parquet exporter package and an independently gated host dataset-
reader capability, so selecting one does not link the other. The old `dynosim`
feature becomes two transport entries in a package owning Dynamo dependencies
and direct-execution implementation; the host owns only its narrow service
traits. `cellular` remains host orchestration. The default distribution requires
HTTP, gRPC, and dry-run transport packages, shipped endpoint packages, and basic
exporters; `full` adds WebSocket, both Dynosim entries, Parquet export, OTLP,
MLflow, and W&B. Host-owned feature choices that affect boundary ABI enter the
host ABI universe; each package's exact private feature set enters only its
artifact-build record. A checked generated matrix maps every legacy feature to
host/core, plugin, and distribution ownership.

Packaging is a required implementation stream. Editable/native install, wheel,
container/Kubernetes image, and uninstall paths include manifests, immutable
artifact closures, allocator provider, authenticated inventory, license/SBOM,
and RECORD/digest entries. Wheel platform tagging considers every native
artifact, not only the executable. Installation and uninstall tests run on
Linux, macOS, Windows x86-64, and Windows arm64.

The static fallback for a component is removed only when all of these are true:

1. its plugin builds independently through the supported SDK command;
2. its manifest and actual registration conform on all supported platforms;
3. existing behavior and artifact suites pass unchanged or with an explicitly
   approved public-contract migration;
4. parent, child, and cellular lock agreement pass;
5. the normative performance gates pass;
6. editing the plugin does not rebuild/relink the host or unrelated plugins;
7. packaging publishes and removes immutable generations atomically on all four
   supported targets;
8. user-facing missing, incompatible, and override diagnostics are covered;
9. import maps prove the allocator, compiled-crate, panic, and native-dependency
   topology and the full ownership conformance suite passes;
10. production searches prove that the migrated ID has no static registry,
    closed-enum, gRPC-binding, exporter-accumulator, or direct-execution path.

No migration step may claim compile-time success by disabling the compiler cache,
changing the optimized profile, weakening LTO in the static baseline, omitting a
required feature from the comparison, or moving work to an unmeasured build
script.

This is a selective reversal of the runtime monocrate consolidation, not a
return to many tightly coupled crates. A boundary is extracted only when it is
publicly reusable or produces a measured build-isolation benefit.

## Verification

- Strict manifest-schema, normalization, and path-hardening tests.
- Complete artifact-closure, embedded build-record, compiler, sysroot, target,
  allocator-provider, native-dependency, and panic-policy mismatch fixtures.
- Priority winner, shadowing, ambiguity, quarantine, and package-transaction
  tests.
- Declared-versus-actual registration conformance tests.
- Trait-object destruction and process-lifetime library-residency tests.
- A third-party exemplar built from a separate Cargo workspace.
- Parent/child re-execution and cellular full-lock agreement/attestation tests.
- Linux x86-64 `.so`, macOS arm64 `.dylib`, Windows x86-64 `.dll`, and Windows
  arm64 `.dll` CI coverage.
- Existing protocol-v2, endpoint, transport, exporter, and mock-server suites.
- Static-baseline versus dynamic-plugin microbenchmarks and end-to-end benchmark
  comparisons.
- Clean and incremental build-time measurements for the host and each
  first-party dependency island.

The SDK ships a minimal plugin example, manifest generator, host-universe and
package-build inspection command, and local conformance harness so third parties can detect
known incompatibility before installation. Documentation never describes this
preflight as proof of native Rust ABI soundness.

### Required conformance fixtures

The separately built exemplar suite contains at least:

- one library registering one endpoint;
- one library registering a transport and endpoint together;
- one library registering multiple exporters;
- one library with both winning and shadowed entries;
- one package that fails after an earlier staged registration, proving rollback;
- one package whose declaration disagrees with its manifest;
- one stale compiler/host-universe package and one tampered package-build record
  that are never called;
- one equal-priority ambiguity across separate packages;
- canonical-versus-alias, alias-versus-alias, redundant-alias, normalization-
  collision, and multiple-package-version fixtures;
- one fully shadowed optional package proving its entry is never called and one
  fully shadowed required package proving it is still activated and validated;
- one winning package that fails during platform loading, proving process poison
  and no lower-priority promotion; one runtime callback error proving no
  promotion after freeze;
- acquisition races that replace the manifest, main library, package-generation
  directory, and private dependency at every acquire/stage/load boundary;
- private dependency tampering, unresolved ambient dependency, ELF/Mach-O loader-
  identity collision, and case-insensitive Windows DLL-basename collision;
- atomic generation install, upgrade, rollback, uninstall, and Windows deferred-
  deletion fixtures;
- locks differing only in private dependency, shadow, quarantine, actual
  descriptor, normalization version, or host build record, all proving unequal
  digests;
- cross-boundary `String`, `Vec`, `Box`, `Arc`, `Rc`, error, trait-object, and
  boxed-future allocate/reallocate/return/drop coverage in both directions for
  every family retained by the final traits, plus import-map proof that every
  shim imports the pinned `mi_*` ABI from the verified `aiperf_alloc_v1`
  provider, runtime relocation targets resolve inside that exact provider under
  preload/interposition fixtures, and every explicit boundary-storage allocation
  obeys the ownership policy;
- subprocess fixtures proving a plugin or host-callback panic aborts and is never
  reported as a caught registration/runtime error;
- a plugin-defined object's `Drop` implementation that records completion before
  process exit, proving object-before-code teardown assumptions while the
  library handle remains retained and without production unload;
- same-process same-lock reuse/different-lock rejection, same-host re-exec, and
  remote-cell lock mismatch fixtures;
- absent transport normalization to HTTP, legacy/open transport Config-v2
  acceptance, mixed-form rejection, neither-exporter acceptance, legacy/open
  exporter mutual exclusion, deterministic legacy exporter order, open-form
  serialization/omission, CLI projection, and protocol removal of
  `transport_typed`;
- a third-party endpoint overriding a first-party endpoint over gRPC unary and
  streaming, proving the old static binding cannot survive;
- Dynosim offline and online through the public direct-execution binding with no
  static ID switch;
- OTLP retain, exact-fold, sketch/folded, sharded, and cellular parity with no
  plugin-specific report type or per-record plugin callback;
- same-/cross-host cellular `ExactRecordsPartitionV1` bounded chunking,
  ordering, digest/count/sequence rejection, and compact-path no-transfer tests;
- cellular `CellCaptureBundleV1` Records/Store projection parity plus missing,
  empty, duplicate, injected, schema, plan-digest, and payload-digest rejection;
- ExactRecords exporter selection forcing explicit reason-tagged retention,
  default exact-fold replacement, and sketch/incompatible-mode rejection;
- best-effort exporter failure and capability-limited artifact-path tests,
  including rejection of unchecked joins;
- help/list/config/profile/eval effect-order tests and plugin-initializer
  conformance checks;
- report and detailed catalog identity for single-process, re-exec, and merged
  cellular execution without absolute paths;
- privileged/user/system/environment discovery-root ownership and mode tests;
- wheel, editable/native, container, default/full, install, upgrade, and
  uninstall discovery tests on all four supported targets; and
- production-code searches proving migrated IDs have no hidden static registry,
  gRPC-binding registry, closed enum, exporter accumulator, or direct path.

Tests MUST use the real dynamic loader and separately produced artifacts. A test
that constructs an extension directly in the test executable proves registry
behavior but does not satisfy loader, linkage, allocator, or ABI conformance.

### Documentation and tooling deliverables

The feature is not complete without:

- generated JSON Schema for `plugins.yaml` schema `2.0`;
- `aiperf plugins list`, `validate`, `inspect-build`, and `lock --output`
  documentation;
- a third-party Cargo template using only allowlisted public crates;
- platform installation layouts and atomic package install/uninstall guidance;
- host-ABI-universe and plugin-artifact-build mismatch troubleshooting guides;
- a priority/override security warning;
- a compatibility table distinguishing source API, host ABI universe identity,
  plugin artifact-build identity, and the absence of a stable/proven Rust ABI;
- benchmark methodology and retained baseline results;
- an updated architecture index and extension-registry record that remove the
  old claim that native AIPerf has no dynamic discovery or dynamic-library seam.
