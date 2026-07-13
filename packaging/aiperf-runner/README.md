<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf runner companion wheel

This platform wheel contains one prebuilt Rust `aiperf-runner` executable. It
does not contain a Python launcher, extension module, inference fallback, or
second CLI. The Python `aiperf` package finds the installed executable through
the companion distribution's wheel RECORD before consulting `PATH`.

Linux x86-64 wheels also contain the source roots for the two mutually
incompatible stock evaluator environments. They are non-script wheel data, not
frontend dependencies or import paths: an exact CPython root, independent NeMo
and OpenBench roots, and the pinned system-library source closure. One strict
versioned registry and the wheel RECORD own the complete payload atomically.
Python passes only those roots to the child; Rust independently verifies every
manifest digest and every upstream distribution RECORD before advertising or
executing either provider. Linux ARM companions intentionally omit this
unsupported inventory and advertise the stable evaluation-unavailable reason.

The trusted native build job supplies two release inputs in `bin/`:

- `aiperf-runner` (or `aiperf-runner.exe`);
- `runner-build.json`, containing the exact domain-separated BLAKE3
  `distribution_id`, 40-digit source revision, `sha256:` Cargo-lock digest, and
  sorted linked Cargo feature list. Offline manifests additionally bind the
  exact `dynamo-aiperf-native` source revision; online manifests carry an empty
  dependency-revision map.

The Linux x86-64 release job additionally creates `evaluator-roots/` from the
two committed `tools/stock_evaluators/*/uv.lock` files. The staging tool accepts
only the verified inventory returned by the canonical stock-manifest generator,
copies regular files into four isolated roots, and writes canonical
`evaluator-roots-v1.json`. The build hook recomputes every staged content-tree
digest and refuses missing, partial, symlinked, or extra payloads.

After staging those inputs, build the wheel:

```bash
cargo build --locked --release -p aiperf-runner
install -m 755 target/release/aiperf-runner \
  packaging/aiperf-runner/bin/aiperf-runner
# Copy the trusted build job's matching runner-build.json beside the binary.
PYTHONPATH=src python -m tools.stage_stock_evaluator_roots \
  --nemo-root /path/to/exact/nemo-prefix \
  --openbench-root /path/to/exact/openbench-prefix \
  --output packaging/aiperf-runner/evaluator-roots
uv build --wheel --out-dir dist/ packaging/aiperf-runner
```

On Windows, copy `aiperf-runner.exe` instead. The build fails unless `bin/`
contains exactly one native ELF, Mach-O, or PE executable and the manifest's
content ID matches its bytes. A Linux x86-64 build also fails without the exact
four-root evaluator payload; an unsupported platform fails if such a payload is
accidentally supplied. The resulting wheel uses a `py3-none-<platform>` tag,
installs the binary directly into the environment's scripts directory, retains
the source/lock/feature/dependency manifest in wheel metadata, and installs the
evaluator roots only as RECORD-owned data.

For a source checkout, stage the same verified inventory beside the selected
development binary instead of under the packaging directory:

```bash
PYTHONPATH=src python -m tools.stage_stock_evaluator_roots \
  --nemo-root /path/to/exact/nemo-prefix \
  --openbench-root /path/to/exact/openbench-prefix \
  --output target/debug/aiperf-runner.evaluator-roots
```

An explicit, `AIPERF_RUNNER_BIN`, or `PATH` runner reads only that adjacent
generated sidecar. An installed companion reads only its own distribution
RECORD. Neither path searches `sys.prefix`, virtual-environment variables, the
ambient evaluator-root variable, or authored Config-v2 filesystem fields.

`AIPERF_RUNNER_WHEEL_PLATFORM_TAG` may be set by a controlled cross-build to
override the host-derived platform tag. The release pipeline must still execute
the binary's `--capabilities` operation on its target architecture before
publishing it.
