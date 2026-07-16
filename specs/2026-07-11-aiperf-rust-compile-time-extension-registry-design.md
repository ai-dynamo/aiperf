<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Rust compile-time extension registry

**Status:** built.

## 1. Decision

Native AIPerf is extensible through ordinary Rust traits, statically linked Cargo
dependencies, and explicit startup registration. It does **not** port Python
AIPerf's `plugins.yaml`, import-string discovery, package scanning, generated
plugin enums, or priority-based replacement.

The build fixes the universe of available implementations. At process startup,
the composition root constructs one `AiperfRegistry`, installs the built-ins,
then applies a deterministic list of linked `AiperfExtension` implementations.
Configuration selects a registered implementation by stable name and receives a
trait object or a fresh trait-object factory product.

Registration therefore happens at startup, but extensibility is compile-time:
an implementation cannot appear unless its crate was compiled and linked into
the executable.

## 2. Why this replaces the Python mechanism

Python needed a data-driven registry because it could import a class named by a
manifest after the package was installed. Rust trait objects do not have a stable
cross-dynamic-library ABI, and Rust has no runtime reflection that enumerates all
`impl Trait for Type` blocks. An `impl` only proves that a type satisfies a
contract; it does not make that type globally discoverable.

The native equivalent separates three concerns:

1. A public trait defines the behavior (`DatasetLoader`, `SamplerFactory`,
   endpoint adapter).
2. A category registry maps a normalized configuration name to a trait object or
   factory.
3. `AiperfExtension::register` adds one linked crate's implementations to the
   aggregate `AiperfRegistry`.

This retains the useful Python property—configuration can select among multiple
implementations—without retaining dynamic imports, schema duplication, runtime
package precedence, or a second source of truth beside the Rust type system.

## 3. Where the aggregate lives

The aggregate contract is the `aiperf_runtime::extensions` module of the `aiperf-runtime`
library crate (`rust/runtime/src/extensions/`). It sits below the runner
executable: the leaf categories it composes (dataset formats, samplers, endpoint
adapters) are sibling modules of the same library crate, and no leaf depends on
the aggregate. The single strict executable, `aiperf-cli`, is the composition
root that constructs, extends, and freezes the aggregate.

```text
extension crate ──▶ aiperf-runtime (extensions module) ──▶ {dataset, endpoints, samplers}
       │                                                     │
       └──────────────── implements leaf traits ─────────────┘

aiperf-cli ─────▶ aiperf_runtime::extensions
```

Keeping the contract below the executable prevents a dependency cycle: a
distribution may add an optional vendor extension dependency to `aiperf-cli`;
that extension depends only on the `aiperf-runtime` library (its extension contract and
the leaf traits it implements), never back on the runner. The application crate
never depends on a vendor extension.

## 4. Public contract

The aggregate owns the registries selected by runtime names:

```rust
pub struct AiperfRegistry {
    dataset_formats: LoaderRegistry,
    samplers: SamplerRegistry,
    endpoints: EndpointRegistry,
    extension_names: BTreeSet<String>,
}

pub trait AiperfExtension {
    fn name(&self) -> &str;
    fn register(&self, registry: &mut AiperfRegistry)
        -> Result<(), ExtensionError>;
}
```

`AiperfRegistry::builtin()` constructs the native distribution. A custom
distribution applies linked extensions explicitly:

```rust
let mut registry = AiperfRegistry::builtin()?;

#[cfg(feature = "acme")]
registry.register_extension(&acme_aiperf_runtime::AcmeExtension)?;

run(registry)
```

An extension registers concrete implementations through typed category APIs:

```rust
impl AiperfExtension for AcmeExtension {
    fn name(&self) -> &str { "acme" }

    fn register(&self, registry: &mut AiperfRegistry)
        -> Result<(), ExtensionError>
    {
        registry.samplers_mut().register(AcmeSamplerFactory)?;
        registry.endpoints_mut().register("acme_chat", AcmeEndpoint)?;
        Ok(())
    }
}
```

The extension crate is trusted native code. This mechanism is not a sandbox or
an ABI boundary.

## 5. Registered categories

The aggregate composes exactly the categories that have runtime name selection
and object-safe factory/lookup seams:

| Category | Stored form | Selection |
|---|---|---|
| Dataset format | paired `Arc<dyn DatasetLoader>` + `Arc<dyn Composer>` | explicit format name or structural probe |
| Sampler | `Arc<dyn SamplerFactory>` | dataset sampling-strategy name |
| Endpoint dialect | adapter descriptor keyed by open `EndpointId` string | authored endpoint name with configured default |

Endpoint identity is an **open string ID** owned by the adapter descriptor
(`EndpointId`), not a closed wire enum of identities compiled into core. A
genuinely new dialect registers its own descriptor and executes as itself, not
as another name for an existing adapter.

Traits selected structurally by application code are injected directly, not put
into a name registry. `Clock`, `RequestSink<R>`, `RequestObserver`,
`SegmentStore`, `RequestMaterializer`, and graph sinks remain ordinary
constructor arguments or generic parameters. Registering them globally would
hide ownership and would not improve configuration selection.

**Accuracy is a directly injected process seam, not a registry entry.**
Canonical dataset preparation, prompt construction, private tests, and grading
belong to one pinned Python/Lighteval worker behind the directly injected
`AccuracyEvaluator` stdio trait (`aiperf_runtime::accuracy_core`). Adding Rust benchmark
or grader factories to `AiperfRegistry` would recreate the duplicated semantics
that boundary removes. External evaluator implementations are selected by
constructing and injecting an `AccuracyEvaluator`, exactly as clocks and
transports are constructor-injected seams rather than runtime-name registry
entries.

Future runtime-name categories—tokenizer factories, arrival factories,
controller policies, reporters/exporters—join `AiperfRegistry` only when their
owning leaf module exposes a typed registry. The aggregate is intentionally not
a string-to-`Any` service locator.

## 6. Factories and state

Registries store an object directly when the implementation is immutable shared
policy (loaders, composers, endpoint adapters, sampler factories). Where a
category instead needs fresh mutable state per selection, it stores a function
pointer rather than a shared object.

Factory registration accepts both a generic `Default` convenience API and an
explicit function-pointer API. The latter supports implementations whose
construction is not `Default` without admitting captured closures or hidden
process state.

## 7. Names, aliases, ordering, and conflicts

- Every category normalizes names at its boundary; lookup never depends on the
  extension's casing.
- Empty names are rejected.
- Built-ins register first.
- Extensions are applied in the caller's explicit order.
- Duplicate extension names are rejected.
- Duplicate canonical names or aliases are rejected. An extension cannot
  silently replace a built-in.
- Dataset auto-detection retains registration order: built-ins first, then the
  explicit extension order. Multiple matching probes remain an ambiguity error.

Python's priority-wins replacement is deliberately absent. Replacement makes
behavior depend on package installation order; a renamed implementation is
clearer and reproducible.

## 8. Atomic extension application

Applying an extension is transactional at registry granularity. The aggregate
clones its small startup-only maps, applies the extension to the clone, and
commits only if every category registration succeeds. If the fifth registration
conflicts, the first four do not leak into the live registry.

This cloning is outside every request/token hot path. Registered implementation
objects are reference-counted; payloads and mutable run state are not copied.

## 9. Runner wiring: one frozen object graph

`aiperf-cli` is the composition root. One fresh runner process builds exactly
one `AiperfRegistry`, applies built-ins and any linked `AiperfExtension`s once,
and freezes the aggregate. `RunnerApplication` freezes that linked registry
together with the runner-owned graph-input resolver, the pair factories, the
compatibility authority, and the protocol-v2 coordinator, so `--capabilities`,
validation, and execution all consume the same exact object graph and the same
frozen product registry.

Creating a fresh built-in category registry inside an execution path would make
linked extensions invisible and is forbidden; execution prepares operations
through the coordinator-owned frozen product registry rather than building a
private one.

The stock workspace binary links no out-of-tree extensions and therefore has the
same behavior and names as before. A vendor or internal distribution adds Cargo
dependencies/features and applies its known extension list in the runner's
composition root. No runtime file or environment variable can add code to an
already-built binary.

## 10. Rejected alternatives

### Port `plugins.yaml`

Rejected. Rust already checks trait conformance and dependency compatibility at
build time; a YAML class path would be unverifiable and cannot instantiate Rust
types without another ABI/reflection mechanism.

### `inventory` / `linkme` self-registration

Rejected. Linker-section collection hides which extensions a distribution
contains, has non-obvious linkage/dead-code behavior, and still requires Cargo
dependencies. An explicit feature-gated list is small, ordered, searchable, and
testable.

### Rust dynamic libraries

Rejected. Ordinary Rust trait-object vtables are not a stable ABI. A true
drop-in runtime plugin system would require a separately versioned C ABI,
`abi_stable`-style boundary, or WASM component contract, including allocation,
panic, async, and version-negotiation rules. That is a different product feature.

### One global mutable registry

Rejected. It makes tests order-dependent, complicates multi-configuration
embedding, and introduces unnecessary synchronization. Registries are owned
values assembled before runtimes and worker threads start.

## 11. Verification gates

1. A test compiled as a separate crate implements a leaf trait and an
   `AiperfExtension`, registers it, and resolves it by name.
2. Duplicate category registrations fail with actionable errors.
3. Duplicate extension names fail.
4. A multi-category extension that fails partway leaves the original aggregate
   unchanged.
5. Built-in catalog counts and aliases remain pinned.
6. Runner tests prove dataset, sampler, and endpoint paths receive the frozen
   aggregate rather than constructing private built-in registries, and that
   capabilities, validation, and execution share one object graph.
7. `cargo fmt`, focused crate tests, workspace tests, and clippy remain green.
