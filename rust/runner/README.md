<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf native runner — source-built wheel (`aiperf-runner`)

This crate is both the `aiperf-runner` binary and its Python wheel. The
co-located `pyproject.toml` uses [maturin](https://www.maturin.rs/) to compile
the executable from source and package it into a platform-specific wheel — the
same layout ai-dynamo uses (`lib/bindings/python/` holds `Cargo.toml` +
`pyproject.toml` together). The wheel contains no Python package, extension
module, or launcher — just the one native executable, installed into the
environment's scripts directory.

The distribution is named `aiperf-runner`: the Python frontend discovers the
executable by that distribution name through its installed wheel RECORD
(`src/aiperf/orchestrator/runner_installation.py`) before consulting `PATH`.

## Requirements

- A working Rust toolchain (`cargo`) matching the workspace `edition = "2024"`.
- The runner's default `dynosim` feature (see `Cargo.toml`) requires the sibling
  `dynamo-aiperf-native/lib/mocker` checkout at build time. The published wheel
  is built **online-only** to stay self-contained (see below).

## Build

```bash
# Release/CI wheel: online-only, manylinux-repaired, self-contained.
# `uv build` reads [tool.uv] config-settings (--no-default-features + auditwheel).
uv build --wheel --out-dir dist/ rust/runner

# Equivalent direct maturin invocation:
maturin build --release --no-default-features --out dist/

# Full runner including the dynosim replay transports (needs the sibling
# dynamo-aiperf-native checkout):
maturin build --release --out dist/
```

`--no-default-features` drops `dynosim`; the resulting runner does not advertise
the offline/online Dynamo replay transports. This matches the base build
described in the top-level `CLAUDE.md`.

The wheel is tagged `<pyver>-none-<platform>` and installs `aiperf-runner` into
the scripts directory of the target environment.

## Editable / develop install

```bash
# From this crate directory. Builds the default (dynosim) runner; add
# --no-default-features for the online-only build without the sibling checkout.
maturin develop --release
```
