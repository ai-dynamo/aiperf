<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Task 3 Report — Verifier Layout Parity

## Changes

- Added `indexResolvedWorldGeometry` to `verify-geometry.ts`. It resolves
  capability layout bottom-up, applies parent-assigned child geometry, translates
  local children into world coordinates, and preserves relative-position lookup
  order to mirror `SceneRenderer.indexSceneNodes`.
- Updated `verifyPackageIr` to use resolved world geometry for box collection,
  connector/fan endpoints and obstacles, dots, and viewport checks.
- Added a focused rail/chip regression proving intrinsic child expansion,
  container reflow, and world-coordinate translation.
- Left `scripts/flow-verifier/geometry.mjs` unchanged. It cannot import the
  TypeScript capability registry without adding a runtime transpilation or
  generated-module harness, so Node verifier parity remains follow-up work.

## Commands and results

- `npm --prefix apps/explainers test -- src/flow/dev-tools/verify-geometry.test.ts`
  - RED: exit 1, `indexResolvedWorldGeometry is not a function`.
  - GREEN: exit 0, 2 tests passed.
- `npm --prefix apps/explainers test`
  - Exit 1: Task 3 tests and the existing layout/renderer tests passed.
  - Rechecks encountered unrelated concurrent changes across managed layout,
    semantic SDK factories, package lowering, and scene resolution; those
    in-progress tests currently fail independently of verifier geometry.
- `npm --prefix apps/explainers run build`
  - Exit 2 due to the same unrelated in-progress files, including incomplete
    managed-layout result types, missing resolution modules, and Node-only test
    type declarations.
- Edited-file diagnostics: no linter errors.
- `git diff --check` for Task 3 files: exit 0.

## Concern

The required full test/build gates are not green because other in-progress,
untracked package tests currently fail independently of verifier geometry.
