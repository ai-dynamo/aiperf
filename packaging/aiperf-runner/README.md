<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf runner companion wheel

This platform wheel contains one prebuilt Rust `aiperf-runner` executable. It
does not contain a Python launcher, extension module, inference fallback, or
second CLI. The Python `aiperf` package finds the installed executable through
the companion distribution's wheel RECORD before consulting `PATH`.

The trusted native build job supplies two release inputs in `bin/`:

- `aiperf-runner` (or `aiperf-runner.exe`);
- `runner-build.json`, containing the exact domain-separated BLAKE3
  `distribution_id`, 40-digit source revision, `sha256:` Cargo-lock digest, and
  sorted linked Cargo feature list. Offline manifests additionally bind the
  exact `dynamo-aiperf-native` source revision; online manifests carry an empty
  dependency-revision map.

After staging those inputs, build the wheel:

```bash
cargo build --locked --release -p aiperf-runner
install -m 755 target/release/aiperf-runner \
  packaging/aiperf-runner/bin/aiperf-runner
# Copy the trusted build job's matching runner-build.json beside the binary.
uv build --wheel --out-dir dist/ packaging/aiperf-runner
```

On Windows, copy `aiperf-runner.exe` instead. The build fails unless `bin/`
contains exactly one native ELF, Mach-O, or PE executable and the manifest's
content ID matches its bytes. The resulting wheel uses a
`py3-none-<platform>` tag, installs the binary directly into the environment's
scripts directory, and retains the source/lock/feature/dependency manifest in
wheel metadata.

`AIPERF_RUNNER_WHEEL_PLATFORM_TAG` may be set by a controlled cross-build to
override the host-derived platform tag. The release pipeline must still execute
the binary's `--capabilities` operation on its target architecture before
publishing it.
