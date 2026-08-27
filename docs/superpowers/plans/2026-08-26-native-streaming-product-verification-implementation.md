<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Product and Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the complete streaming replay surface, prove crash/delivery semantics, run real product E2E, and enforce multi-GB/perpetual boundedness gates.

**Architecture:** Config-v2 remains the primary authoring surface; partial/final results read the checkpoint generation API. Reusable fault matrices exercise contract boundaries, while release-mode ignored tests measure resource slopes without committing large fixtures.

**Tech Stack:** Rust 2024, native `aiperf`, `aiperf-mock-server`, dry-run and E2E harnesses, Linux `/proc` resource sampling, Config-v2 YAML/JSON schema.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at base approval `505efc06b0`, amended by `3fea6f2fe0` and `artifacts/streaming-design/reliability-continuation-course-correction.md`.

## Global Constraints

- Begins only after all foundation, checkpoint/results, pipeline/session, adapter, and cellular implementation plans are merged.
- Every E2E invocation pins a freshly built binary; no harness searches `target/` implicitly.
- Soak data lives under `/mnt/4tb/aiperf-streaming-soak/` and is never committed.
- The progress ledger records exact command, commit, review, invariant, and evidence paths.
- Public defaults are reliability-first: ordinary data, endpoint, checkpoint-attempt, compaction, and exporter faults produce scoped receipts/status and continue or truthfully drain. Config cannot authorize ordinary-fault `FailRun`.

---

### Task V1: Public Config, Capability, Partial-Result, and Documentation Surface

**Files:**
- Modify: `rust/cli/src/flags.rs`
- Modify: `rust/cli/src/load.rs`
- Modify: `rust/cli/src/yaml.rs`
- Modify: `rust/cli/src/lib.rs`
- Modify: `rust/runtime/src/engine/protocol_v2.rs`
- Modify: `rust/runtime/src/engine/application.rs`
- Modify: `rust/runtime/src/config/model/dataset_stream.rs`; Task V1 owns the
  fresh/resume extension to the strict Config-v2 model originally landed by
  foundation Task 3.
- Modify: `rust/runtime/src/config/model/mod.rs`
- Modify: `rust/runtime/src/config/model/config.rs`
- Modify: `rust/runtime/src/config/resolve.rs`
- Modify: `rust/runtime/src/config/validate.rs`
- Create: `rust/cli/src/streaming_results.rs`
- Create: `docs/streaming-datasets.md`
- Test: `rust/cli/tests/streaming_config.rs`
- Test: `rust/cli/tests/streaming_capabilities.rs`
- Test: `rust/cli/tests/streaming_results.rs`
- Extend test: `rust/runtime/tests/streaming_protocol_v2.rs` for Config-v2
  model, resolution, validation, and typed Protocol-v2 run-start projection.
- Create fixture: `rust/cli/tests/fixtures/streaming-shadow.yaml`
- Create fixture: `rust/cli/tests/fixtures/streaming-shadow-resume.yaml`
- Create fixture: `rust/cli/tests/fixtures/streaming-shadow-reliability.yaml`

**Interfaces:**
- Produces: Config-v2-first `dataset_streams`/`shadow_replay`; explicit
  fresh/resume logical-run selection and exact resume locator; feature-accurate
  capability inventory; bounded latest-generation reader output; final/aborted
  metadata; strict reliability policy projection; bounded issue/disposition
  counters and derived-sink status.

```rust
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum StreamingRunStartConfigV2 {
    Fresh,
    Resume { locator: StreamResumeLocatorConfigV2 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamResumeLocatorConfigV2 {
    pub logical_replay_run_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum StreamingRunStartV2 {
    Fresh,
    Resume { locator: StreamResumeLocatorV2 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamResumeLocatorV2 {
    pub logical_replay_run_id: LogicalReplayRunId,
}
```

`StreamingRunStartConfigV2` lives in
`rust/runtime/src/config/model/dataset_stream.rs`, the exact shared Config-v2
model owned by this task. `model/config.rs` carries the authored dataset-stream
section, `model/mod.rs` exports it, `config/resolve.rs` projects it to Protocol
v2, and `config/validate.rs` rejects invalid fresh/resume combinations before
application construction. The loader accepts exactly one canonical checked
lowercase-hex logical ID in its resume locator and projects it to the typed
Protocol-v2 `LogicalReplayRunId`; malformed, missing, or unknown fields fail
before application bootstrap. The checkpoint backend location remains in its
existing config and cannot be redirected by the run locator. `Fresh` carries no
caller-authored ID. `Resume` always carries the exact locator and never means
"latest arbitrary run".

- [ ] **Step 1: Write the RED public-surface test**

Write the six V1 reliability RED cases in the normative amendment below in
this same step, before running Step 2; `streaming-shadow-reliability.yaml` is
referenced before it exists so the intended RED includes missing model/fixture
and result-projection behavior.

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
fn config_projects_exact_fresh_or_resume_run_selection() {
    let fresh = project_profile(include_str!("fixtures/streaming-shadow.yaml")).unwrap();
    assert_eq!(fresh.streaming.run_start, StreamingRunStartV2::Fresh);

    let resumed = project_profile(
        include_str!("fixtures/streaming-shadow-resume.yaml"),
    ).unwrap();
    assert_eq!(
        resumed.streaming.run_start,
        StreamingRunStartV2::Resume {
            locator: StreamResumeLocatorV2 {
                logical_replay_run_id: support::logical_run_id(7),
            },
        },
    );
}

#[test]
fn malformed_or_missing_resume_run_id_fails_config_validation() {
    for authored in [
        support::resume_yaml_without_logical_run_id(),
        support::resume_yaml_with_logical_run_id("not-canonical-hex"),
    ] {
        assert!(validate_and_resolve_profile(&authored).is_err());
    }
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

#[tokio::test(flavor = "current_thread")]
async fn unresolved_resume_refuses_without_allocating_a_replacement_run() {
    let fixture = product_fixture_with_missing_resume_root();
    let requested = fixture.resume_locator_for_run(7);
    let error = fixture.start(StreamingRunStartV2::Resume { locator: requested })
        .await
        .unwrap_err();
    assert!(matches!(error, ProductError::ResumeRunUnresolved { .. }));
    assert_eq!(fixture.logical_run_allocations(), 0);
    assert_eq!(fixture.source_polls(), 0);
    assert_eq!(fixture.endpoint_issues(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn fresh_bootstrap_generation_precedes_source_poll_and_endpoint_issue() {
    let fixture = product_fixture_with_lifecycle_probe();
    let started = fixture.start(StreamingRunStartV2::Fresh).await.unwrap();
    assert_eq!(fixture.logical_run_allocations(), 1);
    assert_eq!(started.resume_locator().logical_replay_run_id, started.run());
    let events = fixture.events().await;
    assert!(events.starts_with(&[
        LifecycleEvent::LogicalRunAllocated,
        LifecycleEvent::BootstrapGenerationCommitted,
        LifecycleEvent::FirstSourcePoll,
        LifecycleEvent::FirstEndpointIssue,
    ]));
    let bootstrap = fixture.bootstrap_generation().await;
    assert!(bootstrap.has_exact_participant_inventory());
    assert_eq!(bootstrap.result_epoch_count(), 1);
    assert_eq!(bootstrap.result_partition_count(), 0);
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-cli --features streaming-s3,cellular,parquet --test streaming_config --test streaming_capabilities --test streaming_results`

Run the shared Config-v2 model/resolution/validation unit tests as well:

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --lib config::
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_protocol_v2
```

- [ ] **Step 3: Implement exact public behavior**

Project strict source/format/session/action/time/order/late/overload/checkpoint/result
values and `StreamingRunStartV2` into Protocol v2. The application resolves the
run before constructing source or endpoint execution: `Fresh` allocates exactly
one `LogicalReplayRunId`, converts it to `StreamRunIdentity`, commits the initial
generation with the exact frozen participant inventory and one canonical empty
result epoch, and returns its exact `StreamResumeLocatorV2`. `Resume` opens only
the locator's run, verifies the run-bound generation/expectations, and reuses
that exact identity. If the locator is absent, malformed, or cannot resolve a
committed generation, refuse before source polling or endpoint issue and do not
call the fresh allocator. A future catalog may resolve the same explicit locator
but may not replace its ID.

Wire the resolved `StreamRunIdentity` into the Task 5E coordinator constructor;
Task 5E accepts only this injected resolved value and performs no config parsing,
resume discovery, fallback allocation, or catalog selection. Expose only
registered descriptors. Add bounded partial-generation rendering and
final/aborted generation metadata; do not add source-specific flag families in
generation 1. Document fidelity, completeness, watermark, overload, restart,
raw/sensitive state, result retention, cellular authority, run-locator handling,
and sizing.

- [ ] **Step 4: Verify green and generated surfaces**

Run Step 2, then from repo root run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
make check-config-schema
pre-commit run check-docs-current --all-files
```

- [ ] **Step 5: Commit**

```bash
git add rust/cli/src/lib.rs rust/cli/src/flags.rs rust/cli/src/load.rs rust/cli/src/yaml.rs rust/cli/src/streaming_results.rs rust/cli/tests/streaming_config.rs rust/cli/tests/streaming_capabilities.rs rust/cli/tests/streaming_results.rs rust/cli/tests/fixtures/streaming-shadow.yaml rust/cli/tests/fixtures/streaming-shadow-resume.yaml rust/cli/tests/fixtures/streaming-shadow-reliability.yaml rust/runtime/src/config/model/dataset_stream.rs rust/runtime/src/config/model/mod.rs rust/runtime/src/config/model/config.rs rust/runtime/src/config/resolve.rs rust/runtime/src/config/validate.rs rust/runtime/src/engine/protocol_v2.rs rust/runtime/src/engine/application.rs rust/runtime/tests/streaming_protocol_v2.rs docs/streaming-datasets.md
git commit -m "feat(cli): expose native streaming replay"
```

## Reliability-Continuation Product Amendment

This section is normative for V1-V6. It does not add a separate implementation
branch: V1 owns public model/protocol/results wiring; V3 owns the reusable fault
matrix; V4 owns real-binary proof; V5 owns sustained-fault boundedness; V6 owns
the evidence ledger.

### V1 Config-v2 and result ownership

Task V1 depends explicitly on foundation Task 1D-R and the downstream
reliability amendments. Add `StreamingReliabilityPolicyV2` to
`rust/runtime/src/config/model/dataset_stream.rs` and its exact resolved
projection to `engine/protocol_v2.rs`:

```rust
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingReliabilityPolicyV2 {
    pub partition_retry_limit: u32,
    pub endpoint_retry_limit: u32,
    pub checkpoint_retry_limit: u32,
    pub export_retry_limit: u32,
    pub retry_backoff_ms: u64,
    pub partition_holes_before_admission_fence: Option<NonZeroU64>,
    pub quarantines_before_admission_fence: Option<NonZeroU64>,
    pub endpoint_failures_before_admission_fence: Option<NonZeroU64>,
    pub checkpoint_failures_before_admission_fence: Option<NonZeroU64>,
}
```

Add the strict bounded result rendering DTOs to
`rust/cli/src/streaming_results.rs`; they consume Task 6B summaries and Task 6D
sink status and are not accepted as authored config:

```rust
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingSinkStatusV2 {
    pub sink_id: String,
    pub generation: u64,
    pub generation_digest: String,
    pub state: StreamingSinkStateV2,
    pub retry_ordinal: u32,
    pub last_issue_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingSinkStateV2 {
    PendingAttempt,
    PendingRetry,
    Exhausted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingReliabilitySummaryV2 {
    pub retry_count: u64,
    pub quarantine_count: u64,
    pub hole_count: u64,
    pub continued_failure_count: u64,
    pub failed_action_count: u64,
    pub is_admission_fenced: bool,
    pub is_degraded: bool,
    pub incomplete_sinks: Vec<StreamingSinkStatusV2>,
}
```

Default retry limits are `3`, `0`, `3`, and `3` for partitions, endpoint
actions, checkpoint attempts, and derived exports respectively; default backoff
is `100` ms. The checkpoint-attempt admission fence defaults to `Some(3)`; the other
three admission-fence thresholds default to `None`. The endpoint threshold is
cumulative committed failed-action receipts; it has no arrival-sensitive reset
or misleading consecutive semantics. Endpoint retry default `0` avoids
uncontrolled target duplication; capability agreement may permit authored
retries only when pre-acceptance or endpoint logical idempotency is proven.
Retry timing uses injected `Clock` and does not enter receipt identity.

Checkpoint retry exhaustion selects `Backpressure` plus admission fencing and keeps a
bounded clock-driven retry owner alive while the truthful prefix drains; it
does not fail the run or advance a generation. A successful later attempt may
release the fence only if no other frozen threshold requires draining.

There is deliberately no authored `fail_run` disposition. Validation rejects
zero backoff with a nonzero retry limit, overflow in duration conversion,
unknown fields, an endpoint retry policy without the required idempotency
capability, and any policy/protocol digest mismatch before source polling or
endpoint issue. The frozen policy digest enters execution-plan and checkpoint
expectations.

Result projection validates `sink_id` with the Task 1D-R component-ID rules,
renders `last_issue_id` as exact lowercase 64-hex BLAKE3, and sorts incomplete
sinks by `(generation, generation_digest, sink_id)`. `is_degraded` is derived
from nonzero hole/quarantine/failed-action/continued-failure counts or admission
fencing, never authored.

Add V1 RED tests:

- `ordinary_fail_run_policy_is_not_authorable`;
- `reliability_policy_unknown_fields_are_rejected`;
- `endpoint_retry_requires_safe_acceptance_or_idempotency`;
- `endpoint_failure_threshold_is_cumulative_and_arrival_order_independent`;
- `reliability_digest_mismatch_refuses_before_poll_or_issue`;
- `partial_results_expose_issue_counts_and_incomplete_sinks`;
- `degraded_run_is_not_rendered_as_failed_run`;
- `continued_failure_sets_degraded`;
- `sink_status_requires_full_generation_identity_and_pending_attempt_zero`.
- `late_input_domain_and_reverse_action_arrival_preserve_domain_local_thresholds`;
- `legacy_v3_is_read_only_and_cannot_precede_or_mix_with_v4`;
- `legacy_v3_participant_state_is_export_only_and_cannot_initialize`;
- `copied_legacy_participant_bytes_cannot_mint_current_v4_context`;
- `raw_action_terminal_or_gap_update_is_not_public_protocol_vocabulary`;
- `host_membership_and_frozen_inventory_views_are_the_only_action_fact_path`;
- `failed_action_prepares_identity_then_finalizes_without_circular_issue_id`;
- `dropped_failed_action_preparation_retries_same_id_without_double_count`;
- `action_retry_backpressure_and_terminal_dispositions_are_type_separated`;
- `dense_action_classification_does_not_hold_reporter_borrow_across_wait`;
- `tombstone_root_extension_requires_same_barrier_reacknowledgement`;
- `tombstone_ack_charges_payload_and_view_without_moving_session_map`;
- `sink_transition_status_cannot_be_authored_without_checked_candidate`;
- `export_failure_receipt_and_durable_output_proof_have_checked_producers`;
- `post_final_export_restart_uses_only_generation_and_derived_status_store`;
- `tampered_embedded_export_receipt_or_counter_refuses_reopen`;
- `first_attempt_and_multi_retry_exhausted_status_reopen_from_independent_counter_authority`;
- `hf_credential_refresh_preserves_pinned_object_identity`.

All lifecycle RED tests remain current-thread Tokio async and await the startup
result. Update `docs/streaming-datasets.md` with scope/class/disposition
semantics, default thresholds, successful-metric exclusion, degraded truthful
completion, admission fencing, and the authority-only terminal boundary.

### V3-V6 fault and evidence ownership

| Product task | Required fault rows and assertion |
|---|---|
| V3 checkpoint/result conformance | transient local/object write/sync, checkpoint capacity, post-CAS notification, compaction, report, and exporter faults retain current head and retry/status; async current-thread sink cases await the sealed transition-candidate CAS, drop execution plus every mutable reporter ledger, then reopen from only the leased final generation and a fresh derived status store; the atomic status reference makes its embedded detailed receipt reachable, while exhausted status independently supplies its checked last-attempt ordinal and counter-before (including first-attempt zero and multi-retry cases); strict restore rejects missing/tampered digest, length, policy, binding, ordinal, or counter authority; illegal/terminal/overflow transitions and crashes before initial status and on both sides of the receipt/status CAS are covered; retained-generation plus frozen-sink reconciliation recovers by full generation identity and exact encoded/parsed charges; memory/local/object open exposes explicit current-v4 versus legacy-v3 leased authority, begin accepts only current-v4, and legacy participant state remains export-only/non-convertible to initialization; foreign run/proof/writer/CAS, impossible cut, or result accounting conflict alone fail-run |
| V4A socket-free | local/HF-compatible partition hole, malformed JSONL record, Baseten session quarantine/tombstone, endpoint terminal failure through the reporter-owned prepare-then-finalize path (including dropped-preparation idempotency), admission fence, forged action terminal/gap refusal, hole→later-valid→checkpoint→resume exact receipt+tombstone-root reachability, late-fragment root invalidation/re-ack, compaction retry, and export incomplete all preserve truthful partial/final results; expired HF credentials refresh against the exact pinned object, while exhausted unchanged identity holes and identity drift fails closed |
| V4B server/cellular | HTTP and gRPC failed action receipts continue later actions; S3 hole continues later objects through the shared A0 credential authority; security/placement digest/ownership/release-proof mismatches remain fail-closed before issue |
| V5A/V5B soak | deterministic injected ordinary faults at a fixed rate do not create positive RSS/task/FD/receipt-index slope; retry, per-input-domain and action sequencer, tombstone, parsed+encoded receipt/status, and sink-supervisor queues remain within item/byte limits; reverse/late domains and scheduled restart do not change thresholds or double-count receipts |
| V6 ledger | every matrix row records exact owner, RED observation, GREEN command, commit, public status, and whether `FailRun` was constructible; any ordinary-fault fail-run is a release blocker |

V3 expands its existing `CheckpointFault::ALL` test with the exact expected
disposition/status table rather than only `complete_or_previous`:

```rust
for case in reliability_fault_matrix() {
    let observed = fixture.run_with_fault(case.fault);
    assert_eq!(observed.disposition, case.expected_disposition);
    assert_eq!(observed.is_run_failed, case.is_authority_truth_or_accounting_invariant);
    assert_eq!(observed.current_generation, case.expected_generation);
    assert!(observed.issue_receipts_are_idempotent());
    assert!(observed.result_and_resume_membership_is_truthful());
}
```

Every V3 case that calls the async status store or retry supervisor is
`#[tokio::test(flavor = "current_thread")] async`, awaits the status/receipt
CAS, drops and reopens the durable store, and asserts the exact full generation,
frozen sink inventory, ordinal, receipt reachability, and unchanged execution
head. A synchronous helper may build cases but may not stand in for awaited
durability or restart.

V4 real-binary output must distinguish:

- `failed`: only a checked terminal-boundary invariant;
- `degraded`: execution continued or truthfully drained with holes,
  quarantines, or failed terminal actions;
- `export_incomplete`: the authoritative native generation is readable while a
  compactor/report/optional exporter is pending or exhausted.

No derived sink failure rewrites an execution outcome or checkpoint head. No
ordinary data fault is hidden: the summary, immutable issue-receipt projection,
and excluded successful membership must agree exactly.

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

Faults include participant/result/index write, file/object sync, pointer rename/CAS, after publication before notification, reader lease loss, GC with reader/prepared lease, compaction, report persistence, safe aborted-generation commit, unsafe shutdown retaining prior partial root, and a long terminal hole excluding bounded provisional completions. For every row assert the exact scoped class/disposition, issue ID/count, admission or sink status, current generation, reachable roots, callback count, horizons, and resumability—not private call order except durability order. Transient/capacity rows retry, backpressure, continue, or expose pending/incomplete sink status; only authority, conflicting-content, frozen-semantic, impossible truthful cut, or accounting rows may observe failed-run shutdown.

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

The strict Dynamo row includes checkpoint-before-profile-bind and checkpoint-
after-bind cases; finite/streaming parity for token IDs, roles, decoded text,
and prefix relationships; and local block-size/profile mismatch refusal. It uses valid
`request.replay` metadata and does not exercise finite-only virtual fallback
hashes.

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

The cellular matrix includes a bound synthesis-profile mismatch and proves zero
prepare acknowledgements, releases, and endpoint issues. This product evidence
lives here, not in V4A's socket-free dry-run fixture; focused C1/C3 unit tests
still pin the same no-early-issue invariant.

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
#[path = "support/streaming_soak.rs"]
mod support;

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

#[test]
#[ignore = "release-mode Dynamo unique-hash slope gate"]
fn dynamo_reconstruction_cache_and_transient_content_are_bounded() {
    let report = support::run_dynamo_unique_hash_soak(support::SoakConfig::from_env()).unwrap();
    assert!(report.cache_high_water_bytes <= report.authored_cache_bytes);
    assert!(report.deferred_descriptor_high_water_bytes <= report.authored_descriptor_bytes);
    assert!(report.token_vector_high_water_bytes <= report.authored_token_bytes);
    assert!(report.decoded_text_high_water_bytes <= report.authored_decoded_text_bytes);
    assert!(report.action_content_high_water_bytes <= report.authored_action_content_bytes);
    assert!(report.rss_slope_bytes_per_input_gib <= 1 * MIB);
    assert!(report.resume_output_is_byte_identical);
    assert_eq!(report.finite_output_digest, report.cache_disabled_output_digest);
    assert_eq!(report.finite_output_digest, report.forced_eviction_output_digest);
    assert_eq!(report.finite_output_digest, report.resumed_output_digest);
    assert!(report.zero_capacity_disabled_cache_construction);
}
```

Run Step 3 before implementing support and record the unresolved-helper RED result.

- [ ] **Step 2: Implement deterministic generation and sampling**

Generate an 8-GiB pinned Hugging Face repository of Baseten Parquet shards with bounded writes and no second resident copy. Serve its exact revision/inventory/ranged shard responses through a bounded local HF-compatible fixture, and execute Config-v2 with source `hf_hub` plus format `baseten_trace`—not a local-source shortcut. Accelerate 24 logical hours with `SimClock`; sample after 10% warmup and at every checkpoint. Emit machine-readable RSS/tasks/FDs/stage items+bytes/session/provisional/index/disk/watermark/schedule/admission/endpoint/drop/duplicate/gap/horizon/cellular-window observations. Enforce the frozen slope, p99, and CPU/action thresholds above.

Also generate a high-cardinality strict Dynamo stream whose hashes do not
repeat. Run once with cache disabled and once with forced small-cache eviction;
both outputs must match finite pure synthesis and checkpoint resume byte-for-
byte. Sample deferred-descriptor, transient token/text, action-content, and
cache capacity independently so an unbounded memoization map or descriptor-to-
decoded-content amplification fails the slope gate.

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
