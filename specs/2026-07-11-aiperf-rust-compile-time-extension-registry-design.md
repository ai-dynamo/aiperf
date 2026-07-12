<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Rust compile-time extension registry

**Status:** decided; implementation accompanies this spec.

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
   `Endpoint`, `AccuracyBenchmark`, `Grader`).
2. A category registry maps a normalized configuration name to a trait object or
   factory.
3. `AiperfExtension::register` adds one linked crate's implementations to the
   aggregate `AiperfRegistry`.

This retains the useful Python property—configuration can select among multiple
implementations—without retaining dynamic imports, schema duplication, runtime
package precedence, or a second source of truth beside the Rust type system.

## 3. Crate boundary

The aggregate contract lives in a small `aiperf-extensions` crate:

```text
extension crate ──▶ aiperf-extensions ──▶ {aiperf-accuracy, aiperf-dataset}
       │                                            │
       └──────────── implements leaf traits ────────┘

aiperf CLI ───────▶ aiperf-extensions
```

It must not live in the `aiperf` application crate. Keeping the contract below
the application prevents this cycle:

```text
aiperf ──optional dependency──▶ vendor-extension ──▶ aiperf
```

With the independent composition crate, a distribution may add an optional
vendor extension dependency to `aiperf`; that extension depends only on the
composition contract and the leaf crates whose traits it implements.

## 4. Public contract

The aggregate owns the registries selected by runtime names:

```rust
pub struct AiperfRegistry {
    dataset_formats: LoaderRegistry,
    samplers: SamplerRegistry,
    endpoints: BuiltinEndpointResolver,
    accuracy: AccuracyRegistry,
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
registry.register_extension(&acme_aiperf::AcmeExtension)?;

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
        registry.accuracy_mut().register_benchmark::<AcmeBenchmark>(&ACME)?;
        Ok(())
    }
}
```

The extension crate is trusted native code. This mechanism is not a sandbox or
an ABI boundary.

## 5. Initial registered categories

The first aggregate includes categories that already have runtime name
selection and object-safe factory/lookup seams:

| Category | Stored form | Selection |
|---|---|---|
| Dataset format | paired `Arc<dyn DatasetLoader>` + `Arc<dyn Composer>` | explicit format name or structural probe |
| Sampler | `Arc<dyn SamplerFactory>` | dataset sampling-strategy name |
| Endpoint dialect | `Arc<dyn Endpoint + Send + Sync>` | authored endpoint name with configured default |
| Accuracy benchmark | `fn() -> Box<dyn AccuracyBenchmark>` | CLI benchmark name/alias |
| Accuracy grader | `fn() -> Rc<dyn Grader>` | CLI grader name/alias |

Traits selected structurally by application code are still injected directly,
not put into a name registry. `Clock`, `RequestSink<R>`, `RequestObserver`,
`SegmentStore`, `RequestMaterializer`, and graph sinks remain ordinary constructor
arguments or generic parameters. Registering them globally would hide ownership
and would not improve configuration selection.

Future runtime-name categories—tokenizer factories, arrival factories,
controller policies, reporters/exporters, and remote accuracy dataset
providers—join `AiperfRegistry` only when their owning leaf crate exposes a
typed registry. The aggregate is intentionally not a string-to-`Any` service
locator.

## 6. Factories and state

Registries store an object directly when the implementation is immutable shared
policy (loaders, composers, endpoint adapters, sampler factories). They store a
function pointer when every selection needs fresh mutable state (benchmarks and
graders).

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
- Accuracy name listings remain lexicographically deterministic.
- Dataset auto-detection retains registration order: built-ins first, then the
  explicit extension order. Multiple matching probes remain an ambiguity error.

Python's priority-wins replacement is deliberately absent. Replacement makes a
benchmark report depend on package installation order; a renamed implementation
is clearer and reproducible.

## 8. Atomic extension application

Applying an extension is transactional at registry granularity. The aggregate
clones its small startup-only maps, applies the extension to the clone, and
commits only if every category registration succeeds. If the fifth registration
conflicts, the first four do not leak into the live registry.

This cloning is outside every request/token hot path. Registered implementation
objects are reference-counted; payloads and mutable run state are not copied.

## 9. CLI wiring

The native CLI constructs `AiperfRegistry::builtin()` once before mode
selection. Accuracy benchmark/grader selection, dataset loading, sampler
construction, and endpoint resolution must use that instance. Creating a fresh
built-in category registry inside an execution path would make linked extensions
invisible and is forbidden.

The stock workspace binary links no out-of-tree extensions and therefore has the
same behavior and names as before. A vendor or internal distribution adds Cargo
dependencies/features and applies its known extension list in its composition
root. No runtime file or environment variable can add code to an already-built
binary.

## 10. Rejected alternatives

### Port `plugins.yaml`

Rejected. Rust already checks trait conformance and dependency compatibility at
build time; a YAML class path would be unverifiable and cannot instantiate Rust
types without another ABI/reflection mechanism.

### `inventory` / `linkme` self-registration

Rejected for the initial implementation. Linker-section collection hides which
extensions a distribution contains, has non-obvious linkage/dead-code behavior,
and still requires Cargo dependencies. An explicit feature-gated list is small,
ordered, searchable, and testable. A future addendum may add a deterministic
descriptor collector if the explicit composition root becomes materially
burdensome.

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
6. CLI tests prove dataset and accuracy paths receive the aggregate rather than
   constructing private built-in registries.
7. `cargo fmt`, focused crate tests, workspace tests, and clippy remain green.

## Addendum — 2026-07-11 (accuracy is a process seam, not a Rust registry)

The accuracy benchmark/grader categories in this design are superseded. Canonical
dataset preparation, prompt construction, private tests, and grading now belong
to one pinned Python/Lighteval worker behind the directly injected
`AccuracyEvaluator` stdio trait. Keeping Rust benchmark or grader factories in
`AiperfRegistry` would recreate the duplicated semantics this boundary removes.

The built `aiperf-extensions` aggregate now composes dataset formats, sampler
factories, and endpoint dialects only. Its dependency on `aiperf-accuracy` and
its `AccuracyRegistry` accessors/error variant are deleted. External evaluator
implementations are selected by constructing/injecting an `AccuracyEvaluator`,
just as clocks and transports are constructor-injected seams rather than
runtime-name registry entries.

## Addendum — 2026-07-11 (the runner is the endpoint composition root)

The compile-time and transactional extension decisions remain authoritative, but the currently
built endpoint category is not yet open enough to satisfy them for a genuinely new dialect. The
closed `EndpointType` wire enum bounds extensions to identities already compiled into core, the
extension proof registers only another name for `ChatEndpoint`, and production execution creates a
fresh built-in registry instead of consuming a registry composed by `aiperf-runner`.

`2026-07-11-aiperf-runner-owned-endpoint-registry-design.md` supersedes that endpoint-specific
shape. Endpoint identity becomes an open string ID owned by the adapter descriptor; the runner
explicitly applies built-ins and linked `AiperfExtension`s once, freezes the aggregate, and uses
that exact value for capabilities, validation, and execution. Dataset-format and sampler extension
semantics in this spec are unchanged. Runtime discovery, linker self-registration, dynamic
libraries, replacement priority, and global mutable registries remain rejected.
