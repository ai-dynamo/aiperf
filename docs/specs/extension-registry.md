<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Extension registry

## Purpose

AIPerf is extensible through ordinary Rust traits, statically linked Cargo
dependencies, and explicit startup registration. The build fixes the universe of
available implementations: an implementation cannot appear unless its crate was
compiled and linked into the executable. Configuration selects a registered
implementation by stable name and receives a trait object or fresh factory
product.

## Built

### Aggregate and categories

`aiperf_runtime::extensions` (`rust/runtime/src/extensions/`) owns the aggregate
`AIPerfRegistry` and the `AIPerfExtension` trait:

```rust
pub trait AIPerfExtension {
    fn name(&self) -> &str;
    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError>;
}
```

`AIPerfRegistry` composes exactly the categories that have runtime name
selection and object-safe factory/lookup seams:

| Category | Stored form |
|---|---|
| Dataset format | `LoaderRegistry` (paired loader + composer, explicit name or structural probe) |
| Sampler | `SamplerRegistry` (dataset sampling-strategy name) |
| Endpoint dialect | `EndpointRegistry` (adapter descriptor keyed by open `EndpointId` string) |
| Exporter | `ExporterRegistry` |
| Actuator | `ActuatorRegistry` |
| Transport | `TransactionalRegistry<Arc<dyn TransportFactory>>` |
| Workload | `TransactionalRegistry<Arc<dyn WorkloadFactory>>` |

`AIPerfRegistry::builtin()` constructs the native distribution by applying the
built-in extensions (`aiperf.builtin.samplers`, `aiperf.builtin.endpoints`,
`aiperf.builtin.exporters`, `aiperf.builtin.actuators`, and the built-in dataset
formats). A custom distribution applies additional linked `AIPerfExtension`s
explicitly in the composition root.

Endpoint identity is an **open string ID** owned by the adapter descriptor
(`EndpointId`), not a closed wire enum. A genuinely new dialect registers its own
descriptor and executes as itself.

### Directly injected seams

Traits selected structurally by application code are injected directly, not
placed in a name registry: `Clock`, `RequestSink<R>`, `RequestObserver`,
`SegmentStore`, request materializers, and graph sinks are constructor arguments
or generic parameters. Accuracy is a directly injected process seam
(`AccuracyEvaluator` in `aiperf_runtime::accuracy_core`), not a registry
category — see [accuracy.md](accuracy.md).

### Registration semantics

- Every category normalizes names at its boundary; empty names are rejected.
- Built-ins register first; extensions apply in the caller's explicit order.
- Duplicate extension names, canonical names, and aliases are rejected — an
  extension cannot silently replace a built-in.
- Dataset auto-detection retains registration order; multiple matching probes
  are an ambiguity error.
- Applying an extension is transactional at registry granularity: the aggregate
  clones its small startup-only maps, applies the extension to the clone, and
  commits only if every category registration succeeds. This cloning is outside
  every request/token hot path; registered objects are reference-counted.

### Frozen object graph

`Application` (`aiperf_runtime::engine::application`) is the composition root. One
runner process builds one `AIPerfRegistry`, applies built-ins and any linked
extensions once, and freezes the aggregate, so the in-process
`aiperf_cli::execute_mode::capabilities_catalog()` API, validation, and execution
derive from the same registered implementation set. Constructing a fresh
built-in registry inside an execution path would make linked extensions
invisible and is forbidden. No runtime file or environment variable can add code
to an already-built binary.

## Source anchors

- `rust/runtime/src/extensions/mod.rs` (`AIPerfRegistry`, `AIPerfExtension`,
  built-in extensions), `rust/runtime/src/extensions/transactional.rs`.
- `rust/runtime/src/engine/application.rs` (`Application` freeze).
- Category registries: `rust/runtime/src/dataset/loader/mod.rs`,
  `rust/runtime/src/dataset/sampler.rs`,
  `rust/runtime/src/endpoints/registry.rs`, `rust/runtime/src/export/mod.rs`,
  `rust/runtime/src/adaptive.rs`, and `rust/runtime/src/engine/registry.rs`.
- Separate-crate extension resolution test in the extensions module tests.
