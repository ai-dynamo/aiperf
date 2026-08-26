<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# E10: Docker verifier transaction unification

## Problem

`rust/runtime/src/eval/execution/docker_process.rs` implements the same Docker
verifier lifecycle in three separate standard-task paths:

- ordinary one-step execution in `execute_with_runtime`;
- NativeGraph execution in the post-callback closure of
  `execute_native_graph_with_runtime`; and
- explicit multi-step execution in `DockerStepSession::run_verifier`.

All three stage declared artifacts for a separate verifier, prepare the
reserved verifier paths, install the selected test tree, execute
`/tests/test.sh`, and read the reward. Their implementations have diverged:
the multi-step path clears `/tests` and `/logs/verifier` after either a
successful or failed shared verifier and combines a cleanup failure with the
primary error, while ordinary and NativeGraph paths do neither. Copying this
transaction again makes policy fixes path-dependent.

Compose has a provider-owned project lease and a different teardown contract;
it is explicitly outside this change. Legacy non-standard packages retain their
compatibility path unchanged.

## Scope and invariants

Create one private Docker verifier transaction used by every standard,
non-Compose Docker path above. It receives explicit already-owned task,
artifact, policy, container-registration, clock, secret, recipe, image, and
test-root inputs. It owns verifier setup, execution, reward acquisition, and
verifier-specific cleanup; outer run/container cleanup remains at its caller.

1. Every in-scope path delegates shared/separate verification to one
   implementation; no caller retains an inline verifier lifecycle.
2. Shared verification always removes `/tests` and `/logs/verifier` after a
   successful, failed, timed-out, or reward-read-failed verifier transaction.
3. A shared cleanup failure joins the primary diagnostic through the existing
   `combine_primary_and_cleanup` contract. A cleanup-only failure is terminal.
4. Separate verification never performs shared-path cleanup. Its fresh staging
   directory, fresh verifier container, artifact-target validation, and
   caller-owned reverse container removal remain intact.
5. One absolute verifier deadline continues to bound staging, create/start,
   workdir inspection, artifact transfer, healthcheck, reserved-path setup,
   test copy, command execution, and reward reading. No operation receives a
   new full timeout.
6. User, environment, workdir, phase-network, secret, artifact exclusion,
   selected test-root, and reward-parsing semantics remain unchanged.
7. NativeGraph preserves adapter ordering: callback success precedes
   collection/verification; callback failure skips verification; backend-owned
   adapter reap ownership remains unchanged.
8. The transaction is private to Docker execution; it does not refactor Compose
   or the legacy compatibility path.

## Design

Introduce a private `DockerVerifierTransaction` (or equivalently scoped helper)
beside `DockerStepSession`. It establishes one optional verifier `Deadline`.
For `Separate`, it materializes/transfers frozen artifacts, registers the
verifier container before creation, then creates, starts, prepares, and checks
that container. For `Shared`, it selects the existing agent container. Both
modes prepare reserved paths, copy the selected test root, prepare shared
workdir state where required, execute the verifier, and read its reward. Only
the shared path invokes `clear_verifier_files`, always combining cleanup with
the primary outcome.

`DockerStepSession::run_verifier` clears its temporary artifact collection
around one helper call. Ordinary and NativeGraph create the same explicit
artifact handoff and invoke the helper instead of retaining their inline copies.

## Test matrix

Tests in `rust/runtime/tests/harbor_docker_runtime.rs` use its recording Docker
runtime and assert observable lifecycle events.

| Case | Required observation |
| --- | --- |
| Ordinary shared success | Exactly one post-verifier reserved-path reset follows success; reward is preserved. |
| Ordinary shared verifier + reset failure | Returned error contains both diagnostics; reset is attempted once after verifier failure. |
| Ordinary separate verifier | No post-verifier shared reset; separate staging/removal remains observable. |
| NativeGraph shared failure | Callback precedes verifier; verifier failure is followed by one reserved-path cleanup before adapter/container cleanup. |
| NativeGraph shared verifier + reset failure | Primary and cleanup diagnostics are joined. |
| NativeGraph separate verifier | No shared reset; adapter ordering and separate isolation remain unchanged. |
| Multi-step regressions | Existing selected-tree reset and fresh-staging tests remain green. |
| Deadline regression | Existing single-step separate-deadline test remains green. |

Implementation is test-first: add ordinary and NativeGraph cleanup assertions,
run them on the pre-change branch for RED, then implement the transaction and
rerun GREEN. Use a unique `/mnt/4tb` `CARGO_TARGET_DIR` for focused and complete
`harbor_docker_runtime` execution. Independent Graham approval is required
before integration.

## Acceptance

All three standard non-Compose Docker paths delegate verification to one
transaction; the test matrix is green after a recorded RED-to-GREEN cycle;
Compose and legacy behavior are unchanged; and independent Graham review has
no unresolved blocker.
