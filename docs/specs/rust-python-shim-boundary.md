<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust/Python shim boundary

## Purpose

Keep the native Rust and Python AIPerf execution products independently usable
while centralizing Python code that exists solely to support native Rust
features. `aiperf profile` remains native Rust execution and `aiperf-python
profile` remains full Python execution. This record defines the narrow,
opt-in Rust-to-Python shim contract; it does not make Python an execution
fallback for the native product, and it does not make Rust an execution
backend for the Python product.

No part of this work imports, merges, cherry-picks, or copies implementation
from `origin/main`. It is a lift-and-shift refactor of code already present on
the branch.

## Built

- `aiperf profile` is dispatched natively by `aiperf-cli`; it does not route
  through the Python profile command.
- `aiperf-python profile` and `python -m aiperf profile` retain a Python
  execution path.
- Rust may delegate Python-owned utility commands such as plotting, analysis,
  plugins, help/completion, and SLURM script generation.
- Some native Rust features currently have Python support implementations,
  including the live-streaming worker and SLURM generation.

## Future requirements

### Product boundary

- Native benchmark/runtime execution must never bootstrap the Python product
  runtime or service mesh.
- Python benchmark/runtime execution must never bootstrap native Rust
  execution.
- Utility delegation is permitted only for an explicit Rust allowlist. It is
  command routing, not a generic execution fallback.
- `aiperf service` must not be reached by a native catch-all Python delegation
  path because it starts the Python service runtime. The native command must
  become native or refuse explicitly.
- Accuracy is out of scope for this refactor because active work owns that
  area.

### Central Python shim package

Create `src/aiperf/rust_shims/` as the central home for Python code that exists
only to support native Rust behavior.

- `rust_shims/__init__.py` stays empty: importing the package cannot initialize
  a shim, register behavior, or create Python-product dependency cycles.
- Shims are executable adapters, not an import API for Rust. Every shim exposes
  a Python `main()` entry point with a documented narrow argv/stdin/stdout
  contract.
- The `aiperf-rust-shim` console script selects from a fixed registered shim
  allowlist, validates its name and arguments, then imports only that selected
  shim and invokes `main()`.
- Rust invokes `aiperf-rust-shim <shim-name> ...` through a subprocess and
  never imports shim modules through PyO3 or depends on Python symbols/types.
- Shim execution is opt-in: installation or import alone cannot activate one.
  Rust calls it only at an explicit feature-owned invocation point.
- Existing Python commands may retain short wrappers that normalize arguments,
  provide help, or invoke a named shim. They must not contain implementation
  logic or rebuild a runtime bridge.

Initial lift-and-shift candidates are:

| Current location | Central destination | Contract owner |
|---|---|---|
| `src/aiperf/post_processors/native_streaming_worker.py` | `src/aiperf/rust_shims/live_streaming_worker.py` | native live-streaming launcher |
| `src/aiperf/cli_commands/slurm/` | `src/aiperf/rust_shims/slurm/` | explicit SLURM-generation utility shim |
| `src/aiperf/config/templates/dynosim_offline_replay.yaml` | `src/aiperf/rust_shims/assets/dynosim_offline_replay.yaml` | Rust embedded-template build input |

`src/aiperf/entrypoint.py` remains outside `rust_shims`: it is concise shared
command-routing infrastructure for the Python product and permitted utility
delegation.

### Python-product preservation

The Python product remains Python-native. Rename native-branded private names
and logging in its bootstrap path so they describe the Python service mesh
accurately, but preserve its complete setup, resolution, and execution
lifecycle. The compatibility result must be behaviorally comparable to the
branch's `origin/main` contract without importing source from that branch.

### Deletion boundary

Do not move code merely because it is adjacent to Rust work. Delete confirmed
unreferenced remnants instead, including the unused Python DynoSim model and
the orphaned Python GPU telemetry worker and its worker-only branch/tests.
Keep active Python/Rust parity tooling as fixture provenance until the project
explicitly freezes or retires its generated fixtures.

### Migration sequence

1. Add the isolated `aiperf-rust-shim` launcher and tests for its allowlist,
   deferred import, argument forwarding, stdio contract, and exit propagation.
2. Lift live-streaming support and SLURM generation into `rust_shims` without
   changing their externally visible contracts; retain only concise compatibility
   wrappers where required.
3. Repoint native Rust invocation sites to the launcher process and exact shim
   names. Replace any in-process Python module/interop dependency at those
   sites.
4. Replace broad native-to-Python command fallback with an explicit utility
   allowlist and refuse or natively implement runtime-owning commands.
5. Preserve and clarify the independent Python profile lifecycle.
6. Delete confirmed dead Python-only Rust remnants and correct stale tool paths.

### Verification

- Python unit tests cover launcher allowlisting, deferred imports, argument
  forwarding, stdio behavior, and process exit propagation.
- Rust integration tests prove each supported native shim is reached only via
  the external launcher command.
- Native profile tests prove Python product runtime bootstrap is not reached.
- Python profile tests prove no Rust executable is launched.
- Existing Rust and Python command behavior remains covered, including allowed
  utility delegation and explicit refusal for unsupported runtime bridging.

## Source anchors

- `rust/cli/src/dispatch.rs` — native command ownership and Python delegation.
- `rust/cli/src/delegate.rs` — current Python command delegation process path.
- `rust/runtime/src/engine/sidecar_input.rs` and
  `rust/runtime/src/engine/live_streaming.rs` — native live-streaming worker
  configuration and launch.
- `src/aiperf/post_processors/native_streaming_worker.py` — current
  native-oriented Python streaming worker.
- `src/aiperf/cli_commands/slurm/generate.py` — current Python SLURM generator.
- `src/aiperf/cli_runner/_single_run.py` — Python-native profile bootstrap.
