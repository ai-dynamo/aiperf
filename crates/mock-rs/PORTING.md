<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Port provenance

`aiperf-mock-rs` was copied wholesale from the deprecated sibling checkout at
`../aiperf-rs/rust/crates/aiperf-mock-rs` while that checkout was clean at
commit `f9456b3d9f69729f398315ca6d85fab39d624fa9`. The byte-for-byte initial import
is preserved in commit `25964b137`.

Every source file, integration test, example, manifest, and tuning helper from
that snapshot is present here. Destination-only changes are intentionally
limited to:

- inheriting this workspace's package metadata and Axum version;
- replacing the removed `aiperf-common` seed helper with canonical
  `aiperf-rng` namespaces;
- updating Axum path-parameter syntax for Axum 0.8;
- resolving the Shakespeare corpus from this workspace instead of an absolute
  path into the deprecated checkout;
- applying the destination Rustfmt, Clippy, Ruff, and SPDX requirements; and
- fixing the imported load-generator example so `--total` is exact when work
  does not divide evenly across workers.

The standalone binary remains outside the product runner dependency graph.
Rust real-network integration fixtures honor `AIPERF_MOCK_RS_BIN` first, then
discover the workspace binary before falling back to `PATH`.

## Verification

The focused port gate is:

```bash
cargo build --release -p aiperf-mock-rs --bins --examples
cargo test --locked -p aiperf-mock-rs --all-targets
cargo clippy --locked -p aiperf-mock-rs --all-targets -- -D warnings
cargo doc -p aiperf-mock-rs --no-deps
cargo machete --with-metadata crates/aiperf-mock-rs
ruff check crates/aiperf-mock-rs/tune_to_trace.py
ruff format --check crates/aiperf-mock-rs/tune_to_trace.py
```

The HTTP transport, graph transport, and scheduled-runtime real-network tests
also spawn this binary to prove it works as an ordinary online inference target.
