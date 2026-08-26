# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sol plan: Origin #59 mmap conversation boundary

**Spec:** `docs/specs/2026-08-26-origin-059-mmap-conversation-boundary.md`

## Decision

Close this port as not applicable after source and ancestry verification.  The
Python mmap fix has no corresponding Rust backend; changing native code would
create a feature rather than port a fix.

## Tasks

1. [x] Inspect `c9288da6c1` and identify the affected Python mechanisms:
   shared mmap cursor, optional prefault, and executor hop.
2. [x] Verify exact upstream ancestry through #60's actual merge rather than
   manufacturing another merge for an already-reachable object.
3. [x] Inspect native dataset ownership and worker materialization.  Confirm
   immutable `Arc`-backed storage and absence of mmap/cursor/executor paths.
4. [x] Run the focused native dataset lookup test.  Do not add a synthetic
   mmap race test because the production mechanism it would exercise does not
   exist.
5. [x] Record the finding, spec, self-Graham result, and campaign closure.

## Guardrail for future work

A future native persistent/mmap dataset feature is a separately designed
feature.  It must not reuse this closure as authority to skip cursor-safety,
startup-prefault policy, or concurrent-reader tests.
