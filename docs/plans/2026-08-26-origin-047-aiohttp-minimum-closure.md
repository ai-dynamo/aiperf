<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sol Plan: Origin/Main 047 aiohttp minimum closure

## Scope

Close upstream `9e96b499d1e942e1b338010fa5d99d9d9c96e074` by proving that its
one-line Python resolver change is already carried by the shared ancestry and
has no native Rust implementation surface.

## Steps

1. Inspect the exact upstream diff and identify its only changed contract:
   `aiohttp>=3.14.3,<4` in `pyproject.toml`.
2. Prove exact ancestry through `52cffc43e2`, including both parent manifests
   and the merged manifest.
3. Inspect native Cargo manifests for transport dependencies and establish that
   aiohttp is not part of the Rust binary.
4. Record findings, specification, tracker closure, and a Graham review of the
   documentation-only closure. Do not add synthetic Rust code or tests.

## Future shared-head integration

This closure branch is based on shared head
`d2c92bae0863328067ca93821533e8f928d3d79c`. Before integration, create a
fresh standalone worktree from the then-current shared head, merge this closure
tip there, resolve only documentation churn, inspect the resulting diff, and
run the documentation checks. The shared branch should receive an actual
two-parent `--no-ff` merge of that reviewed closure tip; it must not attempt to
merge `9e96b499d1` again because that commit is already an ancestor.
