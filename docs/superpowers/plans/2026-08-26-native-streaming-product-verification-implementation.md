<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Product and Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the complete streaming replay surface, prove crash/delivery semantics, run real product E2E, and enforce multi-GB/perpetual boundedness gates.

**Architecture:** Config-v2 remains the primary authoring surface; partial/final results read the checkpoint generation API. Reusable fault matrices exercise contract boundaries, while release-mode ignored tests measure resource slopes without committing large fixtures.

**Tech Stack:** Rust 2024, native `aiperf`, `aiperf-mock-server`, dry-run and E2E harnesses, Linux `/proc` resource sampling, Config-v2 YAML/JSON schema.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at approved commit `505efc06b0`.

## Global Constraints

- Begins only after all foundation, checkpoint/results, pipeline/session, adapter, and cellular implementation plans are merged.
- Every E2E invocation pins a freshly built binary; no harness searches `target/` implicitly.
- Soak data lives under `/mnt/4tb/aiperf-streaming-soak/` and is never committed.
- The progress ledger records exact command, commit, review, invariant, and evidence paths.

---

### Task V1: Public Config, Capability, Partial-Result, and Documentation Surface

**Files:**
- Modify: `rust/cli/src/flags.rs`
- Modify: `rust/cli/src/load.rs`
- Modify: `rust/cli/src/yaml.rs`
- Modify: `rust/cli/src/lib.rs`
- Modify: `rust/runtime/src/engine/protocol_v2.rs`
- Modify: `rust/runtime/src/engine/application.rs`
- Create: `rust/cli/src/streaming_results.rs`
- Create: `docs/streaming-datasets.md`
- Test: `rust/cli/tests/streaming_config.rs`
- Test: `rust/cli/tests/streaming_capabilities.rs`
- Test: `rust/cli/tests/streaming_results.rs`
- Create fixture: `rust/cli/tests/fixtures/streaming-shadow.yaml`

**Interfaces:**
- Produces: Config-v2-first `dataset_streams`/`shadow_replay`; feature-accurate capability inventory; bounded latest-generation reader output; final/aborted metadata.

- [ ] **Step 1: Write the RED public-surface test**

```rust
#[test]
fn normative_streaming_yaml_projects_without_python_or_secrets() {
    let projected = project_profile(include_str!("fixtures/streaming-shadow.yaml")).unwrap();
    assert_eq!(projected.workload.id, "shadow_replay");
    assert_eq!(projected.resources.dataset_streams.items.len(), 1);
    let json = serde_json::to_string(&projected).unwrap();
    assert!(!json.contains("secret-value"));
    assert!(!projected.requires_python());
}

#[test]
fn capability_inventory_is_feature_accurate() {
    let catalog = bootstrapped_catalog();
    assert!(catalog.stream_source.contains_key("local"));
    assert_eq!(catalog.stream_source.contains_key("s3"), cfg!(feature = "streaming-s3"));
    assert_eq!(catalog.stream_checkpoint_backend.contains_key("object_store"), cfg!(feature = "streaming-s3"));
}

#[test]
fn latest_generation_renders_through_bounded_pages() {
    for state in [GenerationState::Partial, GenerationState::Final, GenerationState::Aborted] {
        let fixture = generation_fixture(state, result_page_limit(2));
        let rendered = render_latest_generation(fixture.reader()).unwrap();
        assert_eq!(rendered.state, state);
        assert!(fixture.high_water().index_page_items <= 2);
        assert!(!rendered.contains_provisional_membership());
    }
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-cli --features streaming-s3,cellular,parquet --test streaming_config --test streaming_capabilities --test streaming_results`

- [ ] **Step 3: Implement exact public behavior**

Project strict source/format/session/action/time/order/late/overload/checkpoint/result values into Protocol v2. Expose only registered descriptors. Add bounded partial-generation rendering and final/aborted generation metadata; do not add source-specific flag families in generation 1. Document fidelity, completeness, watermark, overload, restart, raw/sensitive state, result retention, cellular authority, and sizing.

- [ ] **Step 4: Verify green and generated surfaces**

Run Step 2, then from repo root run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
make check-config-schema
pre-commit run check-docs-current --all-files
```

- [ ] **Step 5: Commit**

```bash
git add rust/cli/src/lib.rs rust/cli/src/flags.rs rust/cli/src/load.rs rust/cli/src/yaml.rs rust/cli/src/streaming_results.rs rust/cli/tests/streaming_config.rs rust/cli/tests/streaming_capabilities.rs rust/cli/tests/streaming_results.rs rust/cli/tests/fixtures/streaming-shadow.yaml rust/runtime/src/engine/protocol_v2.rs rust/runtime/src/engine/application.rs docs/streaming-datasets.md
git commit -m "feat(cli): expose native streaming replay"
```

### Task V2: Delivery-Mode and Target-Idempotency Crash Matrix

**Files:**
- Create: `rust/runtime/tests/streaming_delivery_modes.rs`
- No production files. This task consumes the private test support already landed by Tasks 6C1, 6C2, and P4.

**Interfaces:**
- Consumes: checkpoint/pipeline/action contracts.
- Produces: executable semantic proof for `terminal`, `admitted`, `decoded`, `acquired`, and `none`; endpoint-supported logical idempotency keys.

- [ ] **Step 1: Write the RED table-driven matrix**

```rust
#[test]
fn restart_cuts_have_documented_semantics() {
    let cases = [
        (CheckpointDeliveryMode::Terminal, CrashPoint::AfterTargetBeforeCommit, ExpectedReplay::Reissue),
        (CheckpointDeliveryMode::Admitted, CrashPoint::AfterAdmission, ExpectedReplay::DoNotReissue),
        (CheckpointDeliveryMode::Decoded, CrashPoint::AfterDecode, ExpectedReplay::DiagnosticOnly),
        (CheckpointDeliveryMode::Acquired, CrashPoint::AfterAcquire, ExpectedReplay::DiagnosticOnly),
        (CheckpointDeliveryMode::None, CrashPoint::AfterTargetBeforeCommit, ExpectedReplay::NoResumeClaim),
    ];
    for (mode, crash, expected) in cases {
        assert_eq!(delivery_fixture(mode).crash_and_restore(crash).replay(), expected);
    }
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_delivery_modes`

- [ ] **Step 3: Prove the integrated semantic behavior**

Add tests that supported endpoint idempotency derives the key from `(logical_replay_run_id, stable_action_id)` and reports `idempotent_at_least_once_submission`; unsupported endpoint selection fails capability agreement. Crash after target acceptance before commit remains at-least-once without target support. Never claim exactly once. If any row fails because production semantics are missing, stop V2 and reopen a narrowly named owning implementation task with its own RED/GREEN/review commit; do not patch production code from V2.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: every table row and supported/unsupported endpoint case passes.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/tests/streaming_delivery_modes.rs
git commit -m "test(runtime): pin streaming delivery semantics"
```

### Task V3: Checkpoint/Result Failure Conformance

**Files:**
- Create: `rust/runtime/tests/streaming_checkpoint_conformance.rs`
- Extend private fault points in `rust/runtime/src/streaming/checkpoints/local.rs` and `rust/runtime/src/streaming/checkpoints/object_store.rs` only.

- [ ] **Step 1: Write the RED backend matrix**

```rust
fn checkpoint_contract<B: TestCheckpointBackend>(backend: B) {
    for fault in CheckpointFault::ALL {
        let observation = backend.run_with_fault(*fault);
        assert!(observation.current_generation_is_complete_or_previous());
        assert!(observation.uncommitted_objects_are_not_reader_visible());
        assert!(observation.resume_horizon_is_contiguous());
    }
}

#[test] fn memory_backend_conforms() { checkpoint_contract(memory_backend()); }
#[test] fn local_backend_conforms() { checkpoint_contract(local_backend()); }
#[cfg(feature = "streaming-s3")]
#[test] fn object_backend_conforms() { checkpoint_contract(fake_object_backend()); }
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming-s3,parquet --test streaming_checkpoint_conformance`

- [ ] **Step 3: Cover the exact external observations**

Faults include participant/result/index write, file/object sync, pointer rename/CAS, after publication before notification, reader lease loss, GC with reader/prepared lease, compaction, report persistence, safe aborted-generation commit, unsafe shutdown retaining prior partial root, and a long terminal hole excluding bounded provisional completions. Assert current generation, reachable roots, callback count, horizons, and resumability—not private call order except durability order.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: all backend rows pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/tests/streaming_checkpoint_conformance.rs rust/runtime/src/streaming/checkpoints/local.rs rust/runtime/src/streaming/checkpoints/object_store.rs
git commit -m "test(runtime): inject streaming checkpoint failures"
```

### Task V4A: Real-Binary Socket-Free Replay and Restart

**Files:**
- Create: `rust/dry-run-tests/tests/streaming_shadow_replay.rs`
- Create: `rust/dry-run-tests/tests/support/streaming_product.rs`
- Create fixtures: `rust/dry-run-tests/fixtures/streaming/`

- [ ] **Step 0: Build the exact fresh product binary**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo build -p aiperf-cli --release --features streaming-s3,cellular,parquet,grpc`

Record the resulting binary digest; every V4A invocation uses `/mnt/4tb/aiperf-streaming-target/release/aiperf` with that digest.

- [ ] **Step 1: Write and observe the RED process scenario**

```rust
#[path = "support/streaming_product.rs"]
mod support;

#[test]
fn restart_from_checkpoint_matches_sealed_reference() {
    let fixture = support::StreamingProductFixture::local_follow_cross_chunk_graph();
    let first = fixture.run_until_checkpoint_then_kill().unwrap();
    let resumed = fixture.resume(first.generation()).unwrap();
    let sealed = fixture.run_sealed_reference().unwrap();
    assert_eq!(resumed.logical_record_multiset(), sealed.logical_record_multiset());
    assert_eq!(resumed.compacted_metric_store(), sealed.compacted_metric_store());
}
```

Build the feature-matched binary, then run the Step-3 command and record the intended RED result before adding fixtures/helpers.

- [ ] **Step 2: Implement the bounded product fixture**

`StreamingProductFixture` writes partition A, launches the exact `AIPERF_DRY_RUN_BIN` with Config-v2, waits for a committed generation by bounded manifest polling, terminates the process, writes partition B by rename, and resumes with the same checkpoint root. Helpers retain only artifact paths and bounded parsed result pages. Add local finite/follow, JSONL, Baseten/HF-compatible local shards, strict Dynamo, five-minute offset, all delivery restart cuts, cross-chunk conversation/graph, target divergence, partial/final results, and secret/raw-default assertions as table rows.

- [ ] **Step 3: Verify GREEN**

Run: `AIPERF_DRY_RUN_BIN=/mnt/4tb/aiperf-streaming-target/release/aiperf CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-dry-run-tests --test streaming_shadow_replay -- --test-threads=1`

- [ ] **Step 4: Review and commit**

```bash
git add rust/dry-run-tests/tests/streaming_shadow_replay.rs rust/dry-run-tests/tests/support/streaming_product.rs rust/dry-run-tests/fixtures/streaming
git commit -m "test(dry-run): cover streaming replay restart"
```

### Task V4B: HTTP, gRPC, S3-Compatible, and Cellular Product E2E

**Depends on:** Task V4A.

**Files:**
- Create: `rust/e2e-tests/tests/support/streaming_product.rs`
- Create: `rust/e2e-tests/tests/test_streaming_shadow_replay.rs`
- Create: `rust/e2e-tests/tests/test_streaming_cellular.rs`
- Create: `rust/e2e-tests/tests/test_streaming_checkpoint_results.rs`
- Create fixtures: `rust/e2e-tests/fixtures/streaming/`

- [ ] **Step 0: Rebuild and pin the exact product binary**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo build -p aiperf-cli --release --features streaming-s3,cellular,parquet,grpc`

- [ ] **Step 1: Write and observe the RED endpoint/cellular matrix**

Create one parameterized `StreamingServerCase` matrix for HTTP, gRPC, local/S3-compatible source, prepare/release skew, controller/cell restart, and checkpoint-result convergence. Each case launches the exact `AIPERF_E2E_BIN`, the in-repo mock server or bounded fake S3 service, and asserts stable logical membership plus final report order. Run Step 3 before implementing the support module and record the intended unresolved-helper failure.

- [ ] **Step 2: Implement fixed-lifetime test owners**

The support module owns child processes, ports, scratch directories, kill/restart barriers, and bounded artifact readers through RAII. It never searches `target/`, never sleeps for correctness, and always joins children. Reuse the V4A normative configs; add only transport/cellular overlays.

- [ ] **Step 3: Verify GREEN**

Run: `AIPERF_E2E_BIN=/mnt/4tb/aiperf-streaming-target/release/aiperf CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-e2e-tests --test test_streaming_shadow_replay --test test_streaming_cellular --test test_streaming_checkpoint_results -- --test-threads=2`

- [ ] **Step 4: Review and commit**

```bash
git add rust/e2e-tests/tests/support/streaming_product.rs rust/e2e-tests/tests/test_streaming_shadow_replay.rs rust/e2e-tests/tests/test_streaming_cellular.rs rust/e2e-tests/tests/test_streaming_checkpoint_results.rs rust/e2e-tests/fixtures/streaming
git commit -m "test(e2e): cover streaming transports and cellular"
```

### Task V5A: Multi-GiB Runtime Resource Soak

**Files:**
- Create: `rust/runtime/tests/streaming_resource_soak.rs`
- Create: `rust/runtime/tests/support/streaming_soak.rs`

- [ ] **Step 1: Write and observe the ignored RED slope gate**

```rust
#[test]
#[ignore = "release-mode multi-GiB resource gate"]
fn baseten_hf_and_follow_resources_are_bounded() {
    let report = support::run_soak(support::SoakConfig::from_env()).unwrap();
    assert_eq!(report.selected_source_id, "hf_hub");
    assert_eq!(report.selected_format_id, "baseten_trace");
    assert!(report.rss_peak_bytes <= report.baseline_rss_bytes + report.authored_memory_bytes + 256 * MIB);
    assert!(report.rss_slope_bytes_per_input_gib <= 1 * MIB);
    assert!(report.fd_peak <= report.baseline_fds + 2 * report.object_concurrency + 32);
    assert!(report.task_peak <= report.baseline_tasks + report.authored_owner_tasks + 16);
    assert!(report.every_state_high_water_within_authored_budget());
    assert!(report.expected_rate.schedule_slip_p99 <= Duration::from_millis(25));
    assert!(report.expected_rate.cpu_per_action <= report.finite_cpu_per_action * 1.20 + Duration::from_micros(25));
    assert!(report.double_rate.schedule_slip_p99 <= Duration::from_millis(250));
    assert!(report.double_rate.cpu_per_action <= report.finite_cpu_per_action * 1.35 + Duration::from_micros(50));
}
```

Run Step 3 before implementing support and record the unresolved-helper RED result.

- [ ] **Step 2: Implement deterministic generation and sampling**

Generate an 8-GiB pinned Hugging Face repository of Baseten Parquet shards with bounded writes and no second resident copy. Serve its exact revision/inventory/ranged shard responses through a bounded local HF-compatible fixture, and execute Config-v2 with source `hf_hub` plus format `baseten_trace`—not a local-source shortcut. Accelerate 24 logical hours with `SimClock`; sample after 10% warmup and at every checkpoint. Emit machine-readable RSS/tasks/FDs/stage items+bytes/session/provisional/index/disk/watermark/schedule/admission/endpoint/drop/duplicate/gap/horizon/cellular-window observations. Enforce the frozen slope, p99, and CPU/action thresholds above.

- [ ] **Step 3: Verify GREEN**

Run: `AIPERF_STREAM_SOAK_DIR=/mnt/4tb/aiperf-streaming-soak AIPERF_STREAM_SOAK_GIB=8 AIPERF_STREAM_SOAK_LOGICAL_HOURS=24 CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test --release -p aiperf-runtime --features streaming-s3,parquet,cellular --test streaming_resource_soak -- --ignored --nocapture --test-threads=1`

- [ ] **Step 4: Review and commit**

```bash
git add rust/runtime/tests/streaming_resource_soak.rs rust/runtime/tests/support/streaming_soak.rs
git commit -m "test(runtime): add streaming resource soak"
```

### Task V5B: Real-Process Perpetual Soak

**Depends on:** Tasks V4B and V5A.

**Files:**
- Create: `rust/e2e-tests/tests/test_streaming_soak.rs`
- Create: `scripts/streaming-soak.sh`

- [ ] **Step 1: Write and observe RED**

Add an ignored process test that invokes the exact release binary, continuously publishes bounded partitions, resumes across scheduled kills, and validates the V5A thresholds from machine-readable observations. Run Step 3 before the driver script exists and record the intended missing-driver failure.

- [ ] **Step 2: Implement the process driver**

The script validates all required environment values, creates only the explicit scratch subtree, traps child cleanup, streams fixture creation, samples `/proc`, and writes an atomic observation JSON. It never deletes outside its validated scratch path.

- [ ] **Step 3: Verify GREEN**

Run: `AIPERF_E2E_BIN=/mnt/4tb/aiperf-streaming-target/release/aiperf AIPERF_STREAM_SOAK_DIR=/mnt/4tb/aiperf-streaming-soak AIPERF_STREAM_SOAK_GIB=8 AIPERF_STREAM_SOAK_LOGICAL_HOURS=24 CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test --release -p aiperf-e2e-tests --test test_streaming_soak -- --ignored --nocapture --test-threads=1`

- [ ] **Step 4: Review and commit**

```bash
git add rust/e2e-tests/tests/test_streaming_soak.rs scripts/streaming-soak.sh
git commit -m "test(e2e): add perpetual streaming soak"
```

### Task V6: Completion Ledger and Full Branch Gate

**Files:** update `artifacts/streaming-design/implementation-progress.md` only after commands complete.

- [ ] **Step 1: Run formatting and lint**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo fmt --check
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo clippy --workspace --all-targets --features streaming-s3,parquet,cellular,grpc -- -D warnings
```

- [ ] **Step 2: Run library/workspace gates**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming-s3,parquet,cellular,grpc
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-cli --features streaming-s3,parquet,cellular,grpc
```

- [ ] **Step 3: Rerun V4 and V5 against current HEAD**

Record exact binary digest, commands, exit status, elapsed time, and soak observations.

- [ ] **Step 4: Audit every invariant and public field**

For spec invariants 1-38 and every source/format/session/action/backend/placement capability, record exact production file, test name, commit, Graham approval, independent approval, and fresh command evidence. Missing evidence reopens the owning task.

- [ ] **Step 5: Commit the completion record**

```bash
git add -f artifacts/streaming-design/implementation-progress.md
git commit -m "docs: record streaming implementation completion"
```

Do not mark the active goal complete until V1-V6 and every linked subsystem completion gate pass on the same final `HEAD`.
