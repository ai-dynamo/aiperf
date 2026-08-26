# Native vLLM speculative-decode wire implementation plan

> **Author:** Sol
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:test-driven-development for every behavior slice and
> superpowers:verification-before-completion before each success claim.

**Goal:** Move the native vLLM acceptance path to root metrics and dense
histograms while automatically negotiating the trailing usage frame on every
effective streaming chat/completions request.

**Architecture:** Keep the existing engine-neutral observer, metric, and
artifact pipeline. Replace only its vLLM wire adapter, make request negotiation
a post-merge endpoint-body rule, and update the deterministic Rust mock so the
real transport and binary tests exercise the reviewed wire.

**Tech stack:** Rust 2024, serde/serde_json, Hyper/SSE, Axum mock server, native
E2E harness, cargo, sccache.

**Spec:** `docs/specs/2026-08-25-native-vllm-spec-decode-wire.md`

## Global constraints

- Work only in `/mnt/4tb/aiperf-origin-port-052`, based exactly on
  `13fdae44306aa1c04d8ce7a9c71c5c92f53797fe`.
- Begin each production behavior with an observed failing Rust test.
- Use `RUSTC_WRAPPER=sccache`, `SCCACHE_DIR=/mnt/4tb/sccache-port052`, and
  `CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port052`.
- Preserve the #13 neutral metric identities, aggregation, artifact schema, and
  no-stats omission behavior.
- The checkpoint provenance merge must keep its first-parent tree and record
  exact upstream `810fd8bdd40a1c35b64d487b3b8487f0a71a0f6b` as second parent. Root requested
  that merge before implementation; the reviewed native commits descend from
  it. Never cherry-pick or import the upstream Python tree.

---

### Task 1: Root extraction and dense normalization

**Files:**
- Modify: `rust/runtime/src/endpoints/spec_decode.rs`
- Modify: `rust/runtime/src/endpoints/mod.rs`
- Modify: `rust/runtime/src/endpoints/chat_chunk.rs`

**Interfaces:**
- Produces: `extract_vllm_spec_decode_stats(&Value) -> Option<&Value>` for the
  root object and the existing canonical `ObservedSpecDecodeAcceptance`.
- Consumes: vLLM `metrics.speculative_decoding` with dense integer histogram.

- [x] Write root-location and dense worked-example tests, including zero-step,
  fully rejected, detailed, absent width, and obsolete per-choice refusal.
- [x] Run the focused endpoint suite and record the expected root/array RED.
- [x] Replace the extractor and histogram DTO/conversion; remove the obsolete
  streamed-choice capture field and method.
- [x] Add the three highest-value malformed-shape groups: wrong container or
  element type, negative/inconsistent counts, and width mismatch.
- [x] Run the focused suite GREEN and commit the slice.

### Task 2: Effective streaming request negotiation

**Files:**
- Modify: `rust/runtime/src/endpoints/implementation.rs`

**Interfaces:**
- Produces: exact materialized chat/completions bodies with automatically
  negotiated usage frames.
- Consumes: endpoint streaming policy, endpoint extras, latest-turn extras,
  and the existing `BodyPlan` construction path.

- [x] Write failing exact-body tests for no-server-token-count streaming on
  chat and completions.
- [x] Write failing boundary tests for explicit false, null, preserved object
  members, non-object author values, and turn-level `stream: false`.
- [x] Run the endpoint tests and record RED on the missing injection and wrong
  pre-merge gate.
- [x] Implement one small post-merge helper used only by chat/completions.
- [x] Run the focused suite GREEN and commit the slice.

### Task 3: Reviewed mock wire

**Files:**
- Modify: `rust/mock-server/src/handlers.rs`

**Interfaces:**
- Produces: root metrics on non-streaming bodies and empty-choice trailing
  usage frames for both OpenAI endpoints.
- Consumes: the existing `spec_decode_acceptance` mock opt-in and request
  `include_usage` policy.

- [x] Rewrite mock expectations first for dense root metrics on chat and
  completions, streaming and non-streaming, plus streaming opt-out.
- [x] Run the focused mock tests and record the old per-choice RED.
- [x] Move the fixture and serialization to root response/usage structures;
  delete the artificial finish-only stats chunk.
- [x] Run focused mock tests GREEN and commit the slice.

### Task 4: Real transport and product integration

**Files:**
- Modify: `rust/runtime/src/transport/http/sink/endpoint_dispatch.rs`
- Modify: `rust/e2e-tests/tests/test_spec_decode_acceptance.rs`

**Interfaces:**
- Produces: one observer acceptance event from the real empty-choice usage
  frame and complete native profile outputs for chat and completions.
- Consumes: the production HTTP/SSE decoder, endpoint formatter, mock server,
  observer, metric accumulator, and artifact exporters.

- [x] Change the transport integration fixture to root dense metrics on the
  usage frame and run it RED against the old extractor.
- [x] Remove obsolete fast-path capture and run the focused transport test
  GREEN against Task 1.
- [x] Parameterize the product E2E over chat and completions, remove manual
  `stream_options` and `--use-server-token-count`, and retain literal expected
  summary/record/histogram assertions.
- [x] Run the real binary E2E GREEN and commit the slice.

### Task 5: Documentation, verification, review, and closure

**Files:**
- Modify: `docs/reference/spec-decode-acceptance.md`
- Modify: `docs/tutorials/spec-decode-metrics.md`
- Modify: `docs/specs/2026-08-25-native-spec-decode-acceptance-metrics.md`
- Create: `.superpowers/sdd/2026-08-25-native-vllm-spec-decode-wire/graham-review.md`
- Modify: `docs/origin-main-findings/commit-052-810fd8bdd4.md`
- Modify: `docs/porting-origin-main-campaign.md`

**Interfaces:**
- Produces: current public docs, complete verification/review receipts, exact
  upstream ancestry, and an evidence-backed tracker closure.

- [x] Update public docs and the historical #13 supersession marker; run
  documentation link/text checks in the repository's normal verification.
- [x] Run formatting, diff checks, focused runtime/mock/E2E suites, full
  runtime with and without `engine`, CLI, mock-server, and changed-scope Clippy.
- [x] Perform the complete Graham review and second pass over the exact
  base-to-tip diff; repair every finding with TDD and rerun affected tests.
- [x] Commit the self-review receipt and send the exact range to root for an
  independent full Graham review. Repair and re-review until explicitly
  approved.
- [x] After approval, commit closure evidence and tracker state. Re-verify the
  checkpoint merge's parent order, first-parent tree equality, exact second
  parent, clean status, and absence of imported upstream Python changes.
