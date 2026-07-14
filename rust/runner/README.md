<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf native runner — source-built wheel (`aiperf-rust`)

This distribution compiles the Rust `aiperf-runner` executable from source with
[maturin](https://www.maturin.rs/) and packages the resulting binary into a
platform-specific wheel. It contains no Python package, extension module, or
launcher — just the one native executable, installed into the environment's
scripts directory.

It is the compile-from-source counterpart to `packaging/aiperf-runner`, which
instead ingests an already-compiled binary plus a signed build manifest. Use
this wheel when you want `pip wheel` / `maturin build` to drive cargo directly;
use `packaging/aiperf-runner` when a trusted build job produced the binary.

## Requirements

- A working Rust toolchain (`cargo`) matching the workspace `edition = "2024"`.
- By default the runner's `dynosim` feature is enabled (see
  `rust/runner/Cargo.toml`), which requires the sibling
  `dynamo-aiperf-native/lib/mocker` checkout at build time.

## Build

Run from this directory (or point maturin at it):

```bash
# Default: builds the full runner including the dynosim transports.
# Requires the sibling dynamo-aiperf-native checkout.
maturin build --release --manifest-path pyproject.toml --out dist/

# Online-only base runner, no sibling checkout needed:
maturin build --release --no-default-features --out dist/
```

`--no-default-features` drops `dynosim`; the resulting runner will not advertise
the offline/online Dynamo replay transports. This matches the base build
described in the top-level `CLAUDE.md`.

The wheel is tagged `<pyver>-none-<platform>` and installs `aiperf-runner` into
the scripts directory of the target environment. The Python `aiperf` frontend
discovers the installed executable through the companion distribution's wheel
RECORD before consulting `PATH`.

## Editable / develop install

```bash
maturin develop --release
```
