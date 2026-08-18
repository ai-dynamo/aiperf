# Native Harbor adapter filesystem isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve real environment actions while preventing a terminal adapter from mutating verifier inputs.

**Architecture:** A sealed rollout adapter runs in a labelled no-network sidecar with a private source-derived workspace. It uploads a bounded workspace patch with each transition. Rust validates and atomically commits the patch to the verifier workspace before accepting the transition. No terminal or later message can commit another patch.

**Tech Stack:** Rust 2024, Tokio, Docker CLI, strict JSONL, bounded artifact store, `aiperf-runtime`, `aiperf-e2e-tests`.

**Spec:** `docs/superpowers/specs/2026-08-18-native-harbor-adapter-filesystem-isolation-design.md`

## Global Constraints

- Only sealed NativeGraph rollout environment adapters use this path.
- Keep exact authorization, absolute in-image `argv`, empty environment, no-network, finite deadlines, labels, and descriptor-only evidence.
- Patches have only normalized declared relative paths and bounded regular-file content.
- Any patch, adapter, protocol, identity, or cleanup error skips verification.
- Commit complete files through forward-only history.

---

### Task 1: Seal workspace patch authoring

**Files:** `runtime/src/eval/native_graph/{package.rs,rollout_evidence.rs}`, `runtime/tests/{native_graph_package.rs,native_graph_rl_scored.rs}`.

**Interfaces:** Produces immutable `WorkspacePatchLimits`, declared mutable paths, and `FrozenWorkspacePatch { digest, transition_operation, action, observation }` owned by the rollout receipt.

- [ ] Write RED tests: an imported rollout requires positive patch limits and unique normalized paths; an accepted transition has exactly one patch whose action, operation, and observation match.
- [ ] Run `RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_package workspace_patch -- --nocapture` and `native_graph_rl_scored workspace_patch`; expect missing contract/types.
- [ ] Add plan identity material and a private receipt method that accepts the patch descriptor only once before the matching transition.
- [ ] Re-run both tests green; commit full files as `feat(eval): seal Harbor workspace patch evidence`.

### Task 2: Parse and apply patch archives without unsafe filesystem effects

**Files:** Create `runtime/src/eval/native_graph/workspace_patch.rs`; modify `runtime/src/eval/native_graph/{mod.rs,factories.rs}`; create `runtime/tests/native_graph_workspace_patch.rs`.

**Interfaces:** `apply_workspace_patch(root, artifact, limits, mutable_paths) -> Result<ArtifactDigest, NativeGraphWorkspacePatchError>` reads only from the bounded artifact store and returns no payload/path.

- [ ] Write RED tests for a valid `result.txt` patch and unchanged workspace after `../x`, `/work/x`, symlink, device, duplicate path, undeclared path, size overflow, count overflow, and total overflow.
- [ ] Run `RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_workspace_patch -- --nocapture`; expect missing parser/apply API.
- [ ] Parse into a fresh staging directory, reject all invalid entries before destination writes, fsync regular files, then atomically rename only declared paths.
- [ ] Re-run green; commit full files as `feat(eval): apply sealed Harbor workspace patches`.

### Task 3: Make patch commit part of strict transition admission

**Files:** `runtime/src/eval/native_graph/supervision.rs`, `runtime/src/graph/tools/environment_stepper.rs`, `runtime/tests/{native_graph_rl.rs,native_graph_scored_episode.rs}`.

**Interfaces:** Strict `Transition` carries an uploaded workspace-patch reference. The host applies it before the transition is committed to rollout evidence.

- [ ] Write RED tests for missing, foreign-operation, replayed, and post-terminal patch references; each must fail before transition/evidence mutation.
- [ ] Run `RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test native_graph_rl workspace_patch -- --nocapture`; expect no transition field/admission path.
- [ ] Add the strict DTO/state-machine field, require an existing bounded artifact, and invoke the host-owned apply callback before accepting the transition.
- [ ] Re-run `native_graph_rl` and `native_graph_scored_episode` green; commit full files as `feat(eval): commit Harbor transition workspace patches`.

### Task 4: Run rollout adapter in an isolated sidecar

**Files:** `runtime/src/eval/execution/{docker_runtime.rs,docker_process.rs,native_graph_episode.rs}`, `runtime/tests/harbor_docker_runtime.rs`.

**Interfaces:** A Docker rollout lease starts a labelled sidecar from exact authorization, with a private source-derived workspace and full immutable sidecar ID. Task-container workspace is never a sidecar mount.

- [ ] Write RED tests asserting `--network none`, exact labels plus `aiperf.adapter-role=environment`, private-only workspace mount, no verifier-workspace mount, and no attach/kill on label mismatch.
- [ ] Run `RUSTC_WRAPPER= cargo test -p aiperf-runtime --features engine --test harbor_docker_runtime isolated_adapter_ -- --nocapture`; expect current `docker exec` behavior.
- [ ] Add the dedicated builder/spawner/lease. Attach strict stdio and make fence/reap target only the stored sidecar ID. Reap sealed sidecars before collection; leave legacy ordering alone.
- [ ] Re-run green; commit full files as `feat(eval): isolate Harbor environment adapter containers`.

### Task 5: Prove terminal immutability in product execution

**Files:** `rust/e2e-tests/tests/test_harbor_native_graph_rollout.rs`, `docs/specs/native-harbor-agentic-benchmarking.md`, and `llms.txt` if its architecture entry changes.

**Interfaces:** The E2E adapter uploads a `south` patch, sends terminal, then writes `north` to its private workspace. The verifier must receive `south` and reward `0.75`.

- [ ] Write the ignored Docker RED and run it with the current release CLI; expect failure until Tasks 1–4 are wired.
- [ ] Rebuild CLI and run the ignored suite. Assert two selected model calls, no adapter secret/egress, sidecar reaped before collection, verifier then task removal, and reward `0.75`.
- [ ] Update current-truth docs; run `check_agent_files_sync.py` and `check_docs_current.py`; commit full files as `test(eval): prove Harbor verifier artifact immutability`.

### Task 6: Final verification and review

- [ ] Run `native_graph_workspace_patch`, `native_graph_rl`, `native_graph_scored_episode`, and `harbor_docker_runtime` with `--features engine`; run the ignored rollout E2E with the rebuilt CLI.
- [ ] Run `cargo fmt --all --check` and `git diff --check`.
- [ ] Request strict review for archive/path safety, post-terminal commits, sidecar ownership, labels, no secret propagation, and cleanup order.
