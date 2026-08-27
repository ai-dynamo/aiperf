<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Security Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Remove unused direct encryption and zeroization ownership from base
native streaming while preserving deterministic integrity hashing and the
existing TLS/provider/cellular security graph.

**Architecture:** A manifest-policy integration test pins the base feature to
`["engine"]` and rejects direct `chacha20poly1305` or `zeroize` declarations in
the workspace and runtime manifests. The implementation then removes only those
direct declarations and regenerates the lockfile; transitive dependencies are
left to their real consumers.

**Tech Stack:** Rust 2024, Cargo features and lockfile, `toml`.

**Spec:** `artifacts/streaming-design/benchmark-security-scope-course-correction.md`

## Global constraints

- Preserve every BLAKE3 identity, corruption, conflict, CAS, and restart check.
- Preserve provider SDK authentication/TLS and existing cellular admission.
- Do not add a replacement crypto dependency or feature.
- Run Cargo only from `rust/` with the configured wrapper intact and
  `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target`.
- Do not touch the active Task 1D-R runtime implementation files.

### Task 1: Remove direct base-streaming crypto ownership

**Files:**

- Create: `rust/runtime/tests/streaming_dependency_policy.rs`
- Modify: `rust/Cargo.toml`
- Modify: `rust/runtime/Cargo.toml`
- Modify: `rust/Cargo.lock`

**Interfaces:**

- Consumes: the existing `streaming` and `streaming-s3` Cargo features.
- Produces: exact base feature membership `streaming = ["engine"]`; no direct
  workspace/runtime declaration of `chacha20poly1305` or `zeroize`.

- [ ] **Step 1: Add the failing manifest-policy test**

  Parse `rust/runtime/Cargo.toml` and `rust/Cargo.toml` as `toml::Value`. Assert
  that `features.streaming` is exactly the one-element array `engine`; assert
  that neither manifest's relevant dependency table contains
  `chacha20poly1305` or `zeroize`. Name the test
  `base_streaming_owns_no_encryption_or_zeroization_dependency`.

- [ ] **Step 2: Verify RED**

  Run:

  ```bash
  cargo test -p aiperf-runtime --features streaming --test streaming_dependency_policy
  ```

  Expected: the new test fails because base streaming still lists both direct
  optional dependencies.

- [ ] **Step 3: Apply the minimal manifest change**

  Change the runtime feature to `streaming = ["engine"]`; remove the two
  optional runtime dependency entries and the two workspace dependency
  declarations. Let the focused Cargo test regenerate the lockfile minimally
  from the changed manifests; do not run a workspace-wide dependency update.
  Do not attempt to purge transitive
  `zeroize`, TLS, provider SDK, or cellular cryptography.

- [ ] **Step 4: Verify GREEN once**

  Run the Task 1 test plus the existing lightweight and S3 inventory tests,
  followed by `cargo tree -p aiperf-runtime --features streaming` and confirm
  `chacha20poly1305` is absent from that base feature graph. Run exact-file
  rustfmt, targeted Clippy for the new test, and `git diff --check` at the end.

- [ ] **Step 5: Review and commit**

  Review only this task's four-file change and commit it as
  `build(runtime): remove base streaming crypto dependencies`.
