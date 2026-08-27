<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Dataset Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement bounded, resumable native Rust local, Hugging Face, S3, JSONL, Baseten Parquet, and strict Dynamo/NVCF streaming adapters without coupling any source to any format.

**Architecture:** Sources emit immutable partition events and formats decode leased partition access into canonical fragments. Each implementation satisfies the shared conformance harness and checkpoints an exact cursor; the host pipeline owns ordering, sessions, action execution, and results.

**Tech Stack:** Rust 2024, `async_trait(?Send)`, AIPerf streaming contracts, `hf-hub`, Arrow/Parquet, `aws-config = 1.11.0`, `aws-sdk-s3 = 1.144.0`, BLAKE3, bounded blocking executor.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at base approval `505efc06b0`, amended by `3fea6f2fe0` and `artifacts/streaming-design/reliability-continuation-course-correction.md`.

## Global Constraints

- Requires the integrated contract foundation through Tasks 1D-R and 1E plus registry Task 2. Each adapter implements checkpoint participant state from Tasks 5A-R/5B; it does not depend on a concrete checkpoint backend/coordinator.
- Source and format worktrees never edit one another's modules or select one another by concrete type.
- Every acquired/decoded value retains its item+byte lease until incorporation or terminal handoff.
- Blocking file, network-body, Arrow, sort, digest, and catalog work uses `StreamingBlockingExecutor`.
- No adapter calls `SystemTime`, `Instant`, Tokio timers, or Python.
- Cargo commands run from the nested `rust/` workspace; git commands run from the repository root. Tests use `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target`.
- Each task includes the nearest parent module declaration required for its own GREEN build; declaration conflicts are resolved during integration.
- Adapters emit only non-serializable `OrdinaryStreamingIssue` facts plus ordered no-more-before updates through the host reporter and never choose `FailRun`. Record/session facts bind the exact stream and immutable source identity. Adapters release or transfer every owned lease according to the returned fixed-size outcome before decoding the next unit.
- For Tasks A1-A6, Step 1 includes the task-owned RED rows in “Reliability-Continuation Amendment” below, Step 3 implements them, and the existing Step 2/4 command is the one RED/GREEN suite. Do not commit an adapter before its reliability rows pass.

## File Structure

```text
rust/runtime/src/streaming/sources.rs               source built-in registration only
rust/runtime/src/streaming/sources/local.rs         immutable local finite/follow source
rust/runtime/src/streaming/sources/hf_hub.rs        pinned HF inventory and shard source
rust/runtime/src/streaming/sources/hf_catalog.rs    disk-backed sorted HF catalog
rust/runtime/src/streaming/sources/s3.rs            source policy/reconciliation
rust/runtime/src/streaming/sources/s3_client.rs     narrow provider-neutral client trait
rust/runtime/src/streaming/aws.rs               shared AWS SDK client construction
rust/runtime/src/streaming/formats.rs               format built-in registration only
rust/runtime/src/streaming/formats/jsonl.rs         bounded reference JSONL decoder
rust/runtime/src/streaming/formats/baseten.rs       Baseten Parquet decoder
rust/runtime/src/streaming/formats/streaming_dynamo.rs strict Dynamo decoder
rust/runtime/src/graph/recorded/content.rs          shared pure recorded-content synthesis
```

---

### Task A0: Neutral AWS Client Construction

**Depends on:** Foundation Task 0 (`streaming-s3` dependencies).

**Files:**
- Create: `rust/runtime/src/streaming/aws.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Test: `rust/runtime/tests/streaming_aws_client.rs`

**Produces:** one feature-gated `AwsS3ClientFactory` that resolves region, endpoint, proxy, TLS, and credential-provider inputs into a worker-local `aws_sdk_s3::Client`, plus the sole shared `AwsCredentialProviderAuthority` used by S3 source and object-checkpoint consumers. The authority is injected, refresh-capable through the host `Clock`, bounded, and redacts credential material from `Debug`, errors, checkpoints, and provenance. The factory exposes client construction only—no list/get/put/CAS policy—and its `Debug`/errors contain opaque credential-source IDs but never credentials.

- [ ] **Step 1: Write and observe RED**

Add `client_factory_honors_endpoint_and_redacts_credentials` and
`credential_refresh_rebuilds_client_without_changing_source_authority`; run
`CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming-s3 --test streaming_aws_client`.
Feature-off absence is owned by the adapter completion gate's exact
`lightweight_streaming_inventory_excludes_s3` test. Task A6 consumes this
authority and must not define a second AWS credential provider.

- [ ] **Step 2: Implement and verify GREEN**

Construct the SDK config once per prepared worker without global mutable state. Run the Step-1 command and commit:

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/aws.rs rust/runtime/tests/streaming_aws_client.rs
git commit -m "feat(runtime): share bounded AWS client construction"
```

---

### Task A1: Immutable Local Finite/Follow Source

**Files:**
- Create: `rust/runtime/src/streaming/sources.rs`
- Create: `rust/runtime/src/streaming/sources/local.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Test: `rust/runtime/tests/streaming_local_source.rs`

**Interfaces:**
- Consumes: `StreamingDatasetSourceFactory`, `PreparedStreamingDatasetSource`, `StreamingDatasetSource`, `StreamingCheckpointParticipant`, `StreamingBlockingExecutor`.
- Produces: built-in source ID `local`; `LocalSourceConfig`; `LocalSourceCursor { generation, relative_path, object_digest }`.

- [ ] **Step 1: Write the RED contract test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn restore_resumes_after_committed_object_without_duplicates() {
    let fixture = LocalFollowFixture::new(StreamingBudget::items_and_bytes(1, 128)).await;
    fixture.publish_by_rename("000.jsonl", br#"{"id":0}\n"#).await;
    let (mut source, control) = fixture.open().await;
    let first = source.next_event().await.unwrap().into_partition().unwrap();
    let committed = checkpoint_and_notify(&mut source).await;
    drop((first, source, control));

    fixture.publish_by_rename("001.jsonl", br#"{"id":1}\n"#).await;
    let mut restored = fixture.restore(committed).await;
    let next = restored.next_event().await.unwrap().into_partition().unwrap();
    assert_eq!(next.identity().relative_path(), "001.jsonl");
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_local_source`

Expected: FAIL because `LocalSourceFactory` and fixture do not exist.

- [ ] **Step 3: Implement the source contract**

```rust
#[async_trait(?Send)]
impl StreamingDatasetSource for LocalSource {
    fn snapshot(&self) -> &SourceSnapshotReceipt { &self.snapshot }
    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError> {
        self.poll_next_event().await
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for LocalSource {
    fn participant_id(&self) -> CheckpointParticipantId { self.participant_id.clone() }
    async fn checkpoint_view(&mut self, barrier: &CheckpointBarrier)
        -> Result<PreparedParticipantState, CheckpointError> { self.prepare_view(barrier).await }
    async fn initialize(&mut self, state: Option<CommittedParticipantState>)
        -> Result<(), CheckpointError> { self.restore_state(state).await }
    async fn checkpoint_committed(&mut self, receipt: &CommittedParticipantReceipt)
        -> Result<(), CheckpointError> { self.advance_committed(receipt) }
}
```

Acquire directories/files once through private no-follow descriptors. Sort immutable names deterministically. Follow mode accepts publish-by-rename only, remains pending while quiet, emits `Seal` only for finite/explicit authored seal, and refuses mutation after discovery. Run the shared source conformance harness.

- [ ] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS for deterministic order, pending/seal, mutation, cancellation, budgets, leases, and restore.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/sources.rs rust/runtime/src/streaming/sources/local.rs rust/runtime/tests/streaming_local_source.rs
git commit -m "feat(dataset): add bounded local streaming source"
```

### Task A2: Bounded Reference JSONL Format

**Files:**
- Create: `rust/runtime/src/streaming/formats.rs`
- Create: `rust/runtime/src/streaming/formats/jsonl.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Test: `rust/runtime/tests/streaming_jsonl_format.rs`

**Interfaces:**
- Consumes: `StreamingDatasetFormatFactory`, `StreamingDatasetFormat`, `StreamingPartitionDecoder`, `AcquiredPartition`, `DecodeStep`.
- Produces: format ID `jsonl`; strict `JsonlRecordV1`; exact byte/line cursor.

- [ ] **Step 1: Write the RED blocked-output/cursor test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn blocked_output_stops_reading_and_restore_starts_at_next_record() {
    let access = CountingPartitionAccess::jsonl(3, 256);
    let mut decoder = jsonl_decoder(access.clone(), StreamingBudget::items_and_bytes(1, 64));
    let first = decoder.next_batch(DecodeBatchBudget::items_and_bytes(1, 64)).await.unwrap()
        .into_fragment().unwrap();
    assert_eq!(access.completed_reads(), 1);
    let state = decoder.resume_state().unwrap();
    drop(first);
    let mut restored = restore_jsonl_decoder(access, state).await;
    assert_eq!(
        restored.next_batch(DecodeBatchBudget::items_and_bytes(1, 64)).await.unwrap()
            .record_ordinal(),
        1,
    );
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_jsonl_format`

Expected: FAIL because the format is unregistered.

- [ ] **Step 3: Implement strict bounded decoding**

```rust
#[async_trait(?Send)]
impl StreamingPartitionDecoder for JsonlPartitionDecoder {
    async fn next_batch(&mut self, budget: DecodeBatchBudget)
        -> Result<DecodeStep, StreamFormatError> { self.decode_bounded_batch(budget).await }
    fn resume_state(&self) -> Result<DecoderResumeState, StreamFormatError> {
        Ok(self.cursor.resume_state())
    }
}
```

Preserve raw bytes until a complete line, reject unknown fields/oversized lines before proportional allocation, emit one leased `StreamingSessionFragment` at a time, treat partition EOF as `DecodeStep::End` rather than session close, and run the shared format conformance harness.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/formats.rs rust/runtime/src/streaming/formats/jsonl.rs rust/runtime/tests/streaming_jsonl_format.rs
git commit -m "feat(dataset): decode bounded streaming JSONL"
```

### Task A3: Pinned Hugging Face Source and Disk Catalog

**Files:**
- Create: `rust/runtime/src/streaming/sources/hf_hub.rs`
- Create: `rust/runtime/src/streaming/sources/hf_catalog.rs`
- Modify: `rust/runtime/src/streaming/sources.rs`
- Modify: `rust/runtime/src/dataset/loader/public.rs`
- Test: `rust/runtime/tests/streaming_hf_source.rs`

**Interfaces:**
- Produces: source ID `hf_hub`; `HfSourceCursor { repository, commit, subset, split, shard, row_group, row, decoder_digest }`; content-addressed `HfShardCatalog` root and bounded cursor; injected redacted `HfCredentialProvider` whose bounded refresh uses the host `Clock`.

- [ ] **Step 1: Write the RED pinning/catalog test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn prepared_revision_and_complete_inventory_are_immutable() {
    let hub = FakeHub::with_revision("main", "commit-a").with_complete_split(100_000);
    let prepared = prepare_hf(&hub, HfConfig::repo("org/data").revision("main")).await.unwrap();
    hub.move_revision("main", "commit-b");
    let catalog = prepared.catalog().await.unwrap();
    assert_eq!(catalog.commit(), "commit-a");
    assert!(catalog.peak_heap_items() <= catalog.configured_heap_items());
}

#[tokio::test(flavor = "current_thread")]
async fn expired_credential_refresh_retries_the_exact_pinned_object() {
    let fixture = expired_hf_credential_fixture("commit-a", "shard-7");
    let object = fixture.source().next_partition().await.unwrap();
    assert_eq!(object.identity(), fixture.pinned_object("commit-a", "shard-7"));
    assert_eq!(fixture.refreshes(), 1);
    assert!(fixture.debug_and_errors_are_redacted());
}

#[tokio::test(flavor = "current_thread")]
async fn refresh_exhaustion_holes_only_unchanged_immutable_identity() {
    let unchanged = exhausted_refresh_fixture("commit-a", "shard-7");
    assert!(unchanged.next_partition().await.unwrap().is_hole());

    let drifted = exhausted_refresh_with_identity_drift("commit-a", "commit-b");
    assert!(matches!(drifted.next_partition().await.unwrap_err(),
        StreamingFailure::FrozenSemanticDrift { .. }));
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_hf_source`

- [ ] **Step 3: Implement pinned finite acquisition**

Resolve revision once to a commit SHA. Accept only complete static split inventories or API inventories explicitly marked complete; refuse script/generated/heuristic datasets. External-sort bounded runs to a content-addressed disk catalog, acquire immutable cache leases, and run source conformance. Inject a redacted refresh-capable credential provider; on authentication expiry, use bounded `Clock`-driven refresh/backoff and rebuild the request for the exact pinned commit/shard/object identity. Never re-resolve a symbolic revision during refresh. Exhaustion may produce a partition `Hole` only after proving the immutable identity is unchanged; any revision, shard, or content-authority drift is sealed `FrozenSemanticDrift`/`FailRun`. Credential bytes never enter `Debug`, errors, checkpoints, provenance, or issue receipts.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for revision drift, incomplete inventory refusal, million-entry bounded catalog, exact restore, row-limit seal, and unavailable-on-resume.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/sources.rs rust/runtime/src/streaming/sources/hf_hub.rs rust/runtime/src/streaming/sources/hf_catalog.rs rust/runtime/src/dataset/loader/public.rs rust/runtime/tests/streaming_hf_source.rs
git commit -m "feat(dataset): stream pinned Hugging Face shards"
```

### Task A4: Streaming Baseten Parquet Format

**Files:**
- Create: `rust/runtime/src/streaming/formats/baseten.rs`
- Modify: `rust/runtime/src/streaming/formats.rs`
- Modify: `rust/runtime/src/dataset/loader/baseten.rs`
- Test: `rust/runtime/tests/streaming_baseten_format.rs`

**Interfaces:**
- Produces: format ID `baseten_trace`; projected one-pass decoder and exact two-pass disk-index mode; cursor `(object, row_group, row, decoder_digest)`.

- [ ] **Step 1: Write the RED differential test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn streaming_baseten_matches_finite_requests_and_spans_shards() {
    let fixture = baseten_fixture_with_one_session_across_three_shards();
    let finite = finite_baseten_projection(&fixture).await.unwrap();
    let streamed = collect_bounded_fragments(streaming_baseten(&fixture, budget(2, 1 << 20))).await.unwrap();
    assert_eq!(streamed.request_multiset(), finite.request_multiset());
    assert_eq!(streamed.recorded_timing(), finite.recorded_timing());
    assert_eq!(streamed.kv_hints(), finite.kv_hints());
    assert_eq!(streamed.filtered_record_ids(), finite.filtered_record_ids());
    assert_eq!(streamed.recorded_outcomes(), finite.recorded_outcomes());
    assert_eq!(streamed.session_count(), 1);
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,parquet --test streaming_baseten_format`

- [ ] **Step 3: Extract projection and implement bounded decode**

Reuse field/timing/KV/filter/outcome semantics without the finite `Vec<RawRow>` owner. Project before allocation, retain batch lease through incorporation, checkpoint exact row-group/row, and use two bounded passes plus disk index for exact grouping. Register Arrow IPC only if size is rejectable before allocation; otherwise capability validation refuses it. Run format conformance.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for parity, cross-shard session, blocked output, oversized batch refusal, stable two-pass ordering, and cursor restore.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/formats.rs rust/runtime/src/streaming/formats/baseten.rs rust/runtime/src/dataset/loader/baseten.rs rust/runtime/tests/streaming_baseten_format.rs
git commit -m "feat(dataset): stream Baseten parquet traces"
```

### Task A5P: Frozen Shared Recorded-Content Synthesis

**Depends on:** Foundation Tasks 1D-1E.

**Files:**
- Modify: `rust/runtime/src/graph/recorded/content.rs`
- Modify: `rust/runtime/src/graph/recorded/mod.rs`
- Modify: `rust/runtime/src/graph/recorded/trie/mod.rs`
- Modify: `rust/runtime/src/graph/recorded/trie/messages.rs`
- Modify: `rust/runtime/src/dataset/tokenizer.rs`
- Modify: `rust/runtime/src/engine/protocol.rs`
- Modify: `rust/runtime/src/engine/online_execution.rs`
- Test: `rust/runtime/tests/recorded_content_synthesis_profile.rs`

**Produces:** versioned `ContentSynthesisProfileV1`,
`BoundContentSynthesisProfileV1`, `SynthesisAuthority::{Unbound, Bound}`,
explicit `RecordedBlockSamplingAlgorithm`, a tokenizer semantic receipt, and a
cache-free pure block/tail synthesis seam shared by finite and streaming Dynamo.
The profile binds tokenizer artifact/revision/vocabulary/decode/chat-template
semantics, corpus ID and implementation version, content root seed, sampling
algorithm, empty-hash scope, and tail/seed rule version. Diagnostic tokenizer
names are never semantic identity.
The tokenizer receipt also exposes an explicit checked conservative
decoded-byte bound for a requested token count; preparation refuses a tokenizer
whose immutable semantics cannot provide one.

- [ ] **Step 1: Write the RED profile/purity tests**

Add tests proving an environment change after preparation cannot change output,
tokenizer semantic drift is refused, cache-disabled repeated synthesis is
identical, and Python-parity versus BLAKE3-fast is selected only while freezing
the profile. A remote/server tokenizer without an immutable semantic receipt
must fail preparation.

- [ ] **Step 2: Verify RED**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test recorded_content_synthesis_profile`

- [ ] **Step 3: Extract the pure seam**

Separate cache-free sampling from `CorpusContentSynthesizer`'s current mutable
map. Preserve finite byte behavior and default `coding` corpus. Resolve
`AIPERF_WEKA_FAST_CONTENT` once into the frozen algorithm; no synthesis call may
read ambient environment. Preserve authored tokenizer revision through
acquisition and derive a semantic receipt from immutable resolved inputs. Keep
memoization outside the pure seam.
Extract a narrow crate-visible closed-tree message-role planning seam from the
finite future-descendant pass. Both finite Dynamo and P1C must call the same
function; neither `content` nor private `trie::messages` implementation details
become public API.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2 plus existing recorded graph parity tests, then commit only the named
files with `feat(runtime): freeze recorded content synthesis`.

### Task A5: Strict Streaming Dynamo/NVCF Format

**Depends on:** Task A5P.

**Files:**
- Create: `rust/runtime/src/streaming/formats/streaming_dynamo.rs`
- Modify: `rust/runtime/src/streaming/formats.rs`
- Modify: `rust/runtime/src/graph/recorded/dynamo/schema.rs`
- Modify: `rust/runtime/src/graph/recorded/dynamo/mod.rs`
- Modify: `rust/runtime/src/streaming/unit.rs`
- Test: `rust/runtime/tests/streaming_dynamo_format.rs`

**Interfaces:**
- Produces: exactly format ID `streaming_dynamo_trace`, exactly schema
  `dynamo.request.trace.v1`, typed deferred recorded-request fragments retaining
  validated hashes/lengths/request and producer-session identities, response-
  reference metadata, exact record cursor, and checkpointed
  `SynthesisAuthority::{Unbound, Bound}`. Deferred replay material is a typed
  unit variant; hashes must not be disguised as authored conversation text or
  opaque agent payload bytes.

- [ ] **Step 1: Write the RED strictness/cross-object test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn parent_in_later_object_does_not_create_an_early_root() {
    let mut format = dynamo_fixture([child_record("child", "parent"), parent_record("parent")]);
    let child = format.decode_next_object().await.unwrap();
    assert_eq!(child.deferred_requests().len(), 1);
    assert_eq!(child.deferred_requests()[0].parent_id(), Some("parent"));
    let parent = format.decode_next_object().await.unwrap();
    assert_eq!(parent.deferred_requests().len(), 1);
    assert!(parent.deferred_requests()[0].parent_id().is_none());
}

#[test]
fn first_executable_request_binds_block_size_and_later_drift_fails() {
    let mut format = dynamo_fixture([replay_record(16), replay_record(32)]);
    format.decode_next().unwrap();
    assert_eq!(format.decode_next().unwrap_err().code(), "synthesis_authority_mismatch");
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_dynamo_format`

- [ ] **Step 3: Implement separately versioned streaming semantics**

Extract and share finite Dynamo replay parsing, arbitrary-precision hash handling,
alignment, and partial-block removal while keeping finite `dynamo_trace`
byte-compatible. Generation 1 requires valid `request.replay` on every
executable `request_end`; it never invokes the finite complete-trace virtual
fallback allocator. Bind the first executable positive block size before
emitting a deferred executable descriptor, checkpoint unbound/bound authority, and reject
later or restored drift with stable decode codes. Enforce checked bounds on
block size, hash count, token count, and retained deferred-fragment capacity.
Map tool/edge/close records to canonical mutations, treat responses as
reference-only, and never infer root or closure at object EOF.

Cover
repeated/shared hashes; zero, tiny, full, and full-plus-partial inputs; missing
replay; nonzero input with empty hashes; resume before and after binding; record
identity stability across unbound-to-bound authority; and retained trailing-
descendant lookahead facts. A5 never releases an action: P1C waits for a typed
whole-producer-tree closure receipt before invoking the shared future-aware
message-role pass. Indefinite shapes without a configured closure proof are
refused by P1C using the stable unbounded-causality session failure; A5 only
retains the typed parent/descendant facts needed to decide later.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for strict parsing, deferred descriptor geometry,
duplicate idempotency/conflict, stable error codes, block boundaries, authority
resume, identity stability, and retained parent/descendant closure facts.
Materialized finite/streaming token/message/text/prefix parity and no-early-
release evidence belong to P1C after whole-tree closure exists.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/formats.rs rust/runtime/src/streaming/formats/streaming_dynamo.rs rust/runtime/src/streaming/unit.rs rust/runtime/src/graph/recorded/dynamo/schema.rs rust/runtime/src/graph/recorded/dynamo/mod.rs rust/runtime/tests/streaming_dynamo_format.rs
git commit -m "feat(dataset): decode strict streaming Dynamo traces"
```

### Task A6: Native S3 Finite/Follow Source

**Depends on:** Task A0.

**Files:**
- Create: `rust/runtime/src/streaming/sources/s3_client.rs`
- Create: `rust/runtime/src/streaming/sources/s3.rs`
- Modify: `rust/runtime/src/streaming/sources.rs`
- Test: `rust/runtime/tests/streaming_s3_source.rs`

**Interfaces:**
- Produces: feature-gated source ID `s3`; narrow `S3Client`; lossless manifest/no-backfill and explicitly lossy reconciliation policies; version/ETag/size/BLAKE3 identity.

- [ ] **Step 1: Write the RED reconciliation test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn notification_loss_and_late_key_are_recovered_before_interval_seal() {
    let s3 = FakeS3Client::new();
    s3.put_versioned("bucket", "0002", b"two", "v2");
    let mut source = prepare_s3(&s3, lossless_manifest_policy()).await.unwrap();
    let _ = source.next_event().await.unwrap();
    s3.put_versioned_without_notification("bucket", "0001", b"one", "v1");
    s3.seal_manifest(["0001", "0002"]);
    assert_eq!(drain_partition_keys(&mut source).await.unwrap(), ["0001"]);
}

#[tokio::test(flavor = "current_thread")]
async fn pagination_and_identity_rules_are_explicit() {
    let s3 = FakeS3Client::with_page_size(2);
    s3.put_versioned("bucket", "v", b"one", "version-1");
    s3.put_unversioned("bucket", "u", b"two", multipart_etag());
    let observed = collect_s3(s3, reconciliation_budget(2, 4096)).await.unwrap();
    assert_eq!(observed[0].identity().provider_version(), Some("version-1"));
    assert!(observed[1].identity().content_digest().is_some());
    assert_ne!(observed[1].identity().content_digest_text(), multipart_etag());
    assert!(observed.high_water().list_page_items <= 2);
}

#[test]
fn lossless_and_lossy_policies_fail_or_label_honestly() {
    assert!(matches!(validate_s3_policy(mutable_listing_without_hard_no_backfill()),
        Err(StreamSourceError::LosslessFrontierUnprovable { .. })));
    let lossy = validate_s3_policy(authored_lossy_window(128)).unwrap();
    assert_eq!(lossy.fidelity(), SourceFidelity::LossyWindow { max_keys: 128 });
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming-s3 --test streaming_s3_source`

- [ ] **Step 3: Implement provider-neutral policy and AWS transport**

```rust
#[async_trait]
trait S3Client: Debug + Send + Sync {
    async fn list_page(&self, request: S3ListRequest) -> Result<S3ListPage, S3ClientError>;
    async fn get_version(&self, request: S3GetRequest) -> Result<S3ObjectBody, S3ClientError>;
}
```

Notifications are hints; reconciliation is authority. Add named mutation cases `listing_changes_between_pages_reconcile_without_false_frontier`, `versioned_identity_survives_overwrite`, `unversioned_overwrite_is_refused`, and `hard_no_backfill_violation_fails_before_seal`. Pagination never advances a frontier. Lossless requires sealed manifest/time bucket or immutable monotonic keys plus hard no-backfill; otherwise retain one authored bounded window and label output lossy. Multipart ETag is never a digest. Backoff uses injected `Clock`; credentials/signed URLs are non-serializable and redacted. Run source conformance.

- [ ] **Step 4: Verify green and feature-off absence**

Run Step 2. Expected: pagination, versioned/unversioned identity, multipart ETag, listing mutation, notification loss, hard-no-backfill, lossy labeling, and bounded reconciliation cases pass. The adapter completion gate separately runs `lightweight_streaming_inventory_excludes_s3` under `--no-default-features --features streaming` to prove feature-off absence.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/sources.rs rust/runtime/src/streaming/sources/s3_client.rs rust/runtime/src/streaming/sources/s3.rs rust/runtime/tests/streaming_s3_source.rs
git commit -m "feat(dataset): follow immutable S3 partitions"
```

## Reliability-Continuation Amendment

Tasks A1-A6 consume the exact Task 1D-R `StreamingIssueReporter`; the reporter
is injected in source/format prepare contexts and remains host-owned. No adapter
implements its own threshold counter or sleep loop. Existing task suites gain
the following RED cases before their production step, then pass in the same one
task GREEN invocation already specified above.

| Owner | Fault and scope | Required disposition/effect | RED tests added to the owner's existing suite |
|---|---|---|---|
| A1 local source | an opened immutable partition disappears before bounded acquisition | retry, then partition hole; later rename-published partitions continue | `missing_partition_becomes_checkpointed_hole_and_follow_continues` |
| A2 JSONL | isolated syntax, schema, or oversized line with a recoverable next boundary | record quarantine; release bytes through the quarantined receipt and decode the next line | `malformed_jsonl_record_is_quarantined_without_aborting_partition`, `quarantine_restores_without_double_count` |
| A3 HF | transient catalog/shard authentication or read failure | credential refresh/read retry under `Clock`, then partition hole only if the frozen immutable identity is still known | `hf_shard_retry_exhaustion_records_hole_and_continues` |
| A3 HF | revision/inventory/object identity, length, or digest changes after freeze; no authorized immutable source can be acquired without substituting identity | invariant issue and host-validated fail-run before decode | `hf_identity_or_authority_drift_never_degrades_to_hole` |
| A4 Baseten | invalid row isolated to one record or recoverable session | record/session quarantine, deterministic state retirement, continue later row groups | `bad_baseten_row_quarantines_only_its_record`, `bad_trace_session_retires_state_and_next_session_continues` |
| A4 Baseten | prepared Arrow/Parquet schema or projection changes | frozen-semantic invariant, fail-run before mixed-schema emission | `baseten_schema_drift_is_terminal_before_fragment_output` |
| A5 Dynamo | malformed request/replay metadata confined to one producer tree | record/session quarantine after the exact closure proof; no guessed reconstruction | `invalid_replay_tree_is_quarantined_and_neighbor_tree_continues` |
| A5 Dynamo | tokenizer, synthesis-profile, block-size, or bound semantic digest changes | frozen-semantic invariant and fail-run | existing profile-mismatch tests plus `bound_profile_drift_cannot_be_quarantined` |
| A6 S3 | list/get throttling, notification loss, or ranged-read timeout | bounded retry, reconciliation, then partition hole with later object continuation | `s3_retry_exhaustion_records_hole_and_reconciliation_continues` |
| A6 S3 | version/ETag/size/digest differs from the frozen object identity; authorization can succeed only by selecting another object | identity invariant and fail-run | `s3_identity_substitution_is_refused_not_holed` |

The source/format conformance cases now require these observations:

```rust
#[tokio::test(flavor = "current_thread")]
async fn ordinary_adapter_faults_leave_the_stream_live() {
    for case in continuation_adapter_cases() {
        let report = case.run_fault_then_valid_unit().await;
        assert_eq!(report.issue_receipts, 1);
        assert_eq!(report.valid_units_after_fault, 1);
        assert!(!report.is_run_failed);
        assert!(report.all_item_and_byte_leases_settled());
    }
}

#[tokio::test(flavor = "current_thread")]
async fn adapter_issue_receipt_is_stable_across_checkpoint_resume() {
    let first = jsonl_fault_fixture().run_until_issue_then_checkpoint().await;
    let resumed = jsonl_fault_fixture().resume(first.generation()).await;
    assert_eq!(resumed.issue_id, first.issue_id);
    assert_eq!(resumed.threshold_count, 1);
}
```

Partition holes advance only the source-acquisition frontier represented by the
typed hole receipt and same-generation `HandledIssueCut`; they never fabricate decoded/session/action progress.
Quarantine advances only through the exact record/session boundary proven by
the decoder/coordinator and keeps the unit out of successful logical metric
membership. A missing boundary that prevents truthful continuation escalates
to the ordering/cut invariant owner; the adapter does not guess a cursor.

Every record/session issue and no-more-before update carries the same checked
`StreamingInputDomainIdentity`. Add `same_record_id_in_two_sources_has_distinct_issue_id`
and `hole_then_later_valid_checkpoint_resume_keeps_receipt_reachable`; the latter
must refuse restore when either handled-cut root or receipt partition is absent.

Transient source authentication may refresh and retry without changing the
frozen object identity. Capability/credential refusal discovered before the run
remains preflight failure. During a run, fail-run is allowed only when no
authorized immutable source can be acquired without falsifying the already
frozen identity; it is never selected merely because one refresh attempt failed.

## Adapter Completion Gate

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming-s3,parquet --test streaming_reliability --test streaming_contract_conformance --test streaming_local_source --test streaming_jsonl_format --test streaming_hf_source --test streaming_baseten_format --test streaming_dynamo_format --test streaming_s3_source
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --no-default-features --features streaming --test streaming_feature_inventory lightweight_streaming_inventory_excludes_s3 -- --exact
```

Every adapter must pass Graham and independent review with no source×format branching, unbounded inventory, blocking hot-path work, leaked secret, or cursor ambiguity.
