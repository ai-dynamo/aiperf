# Native Per-Chunk Usage and Bundled-First-Chunk ITL

## Status

Implemented design for origin/main tracker #51, exact upstream commit
`324bb05773b3f99743c6516018f3c30cfe33de0b`; independent review is the sole
remaining closure gate.

## Purpose

Correct inter-token latency and output-token throughput per user when a streaming
chat server emits multiple generated tokens in the first content chunk. The
feature is explicitly opted into because `continuous_usage_stats` is a serving
engine extension rejected by strict OpenAI endpoints.

## Authoring contract

`endpoint.per_chunk_usage` defaults to false and is exposed as
`--per-chunk-usage[=true|false]`. It is legal only when all of these are true:

- endpoint type is exactly `chat`;
- streaming is enabled;
- `use_server_token_count` is enabled.

Validation applies to the default endpoint and every named endpoint profile
before execution. The flag-only and YAML-plus-CLI paths use identical
precedence: an absent CLI flag preserves YAML; an explicit boolean overlays it.

## Request contract

For a valid opted-in endpoint, chat payload construction ensures these defaults:

```json
{"stream_options":{"include_usage":true,"continuous_usage_stats":true}}
```

Existing object keys and explicit values win. In particular, explicit
`include_usage: false` or `continuous_usage_stats: false` are not overwritten,
matching upstream `setdefault` behavior. A non-object authored `stream_options`
value is left unchanged rather than replaced. Without `per_chunk_usage`, the
existing server-token-count behavior adds only `include_usage`.

## Response and measurement contract

The HTTP response reducer examines parsed responses in wire order. It continues
to reconcile terminal usage from the latest reported value. Separately, when the
prepared endpoint is opted in, it retains the cumulative
`usage.completion_tokens` from the first meaningful content response—the same
response that releases TTFT. Role-only, empty, finish-only, and usage-only frames
cannot supply this count. Reasoning content is meaningful content, so its
cumulative completion count is valid and already shares OSL's completion-token
unit (visible output plus reasoning).

The count crosses the transport-neutral observer boundary as an optional usage
fact and is retained in `TokenCounts` on `RecordIngest`. It remains absent when
the option is off, even if a server volunteers usage on content chunks.

For a valid request record:

```text
decode_duration = request_latency - time_to_first_token
first_chunk = reported_first_chunk when 0 < reported_first_chunk < OSL else 1
inter_token_latency = decode_duration / (OSL - first_chunk)
output_token_throughput_per_user = 1 second / inter_token_latency
```

Missing usage preserves the existing `OSL - 1` behavior. A reported zero or a
count greater than or equal to OSL falls back to one and emits a process-bounded
warning rather than suppressing the metric. The main metrics accumulator and
adaptive window sampler use one shared divisor helper so event and record paths
cannot drift.

## Mock and E2E contract

The public AIPerf option remains chat-only, matching upstream validation. The
Rust mock itself supports both OpenAI streaming surfaces,
`/v1/chat/completions` and `/v1/completions`, because it is also a standalone
wire-compatible test target. Both request models accept the vendor stream option
and a test-only first-chunk bundle size.

`continuous_usage_stats` and `include_usage` are independent. Continuous true
adds the cumulative three-field usage object (`prompt_tokens`,
`completion_tokens`, `total_tokens`) to every generated reasoning or visible
content frame. Absent or explicit false adds no usage to those frames.
`include_usage` alone controls the terminal empty-choice full-usage frame.
Bundling joins the first N visible output tokens into one content frame and
retains one-token frames afterward. Timed and fast streams share this contract;
frames emitted before an injected mid-stream error retain cumulative usage, but
an error never synthesizes a terminal usage frame.

The native-binary E2E test must exercise the real CLI/config resolver, endpoint
formatter, HTTP/SSE stack, response reducer, metrics observer, accumulator, and
artifact exporter. It proves both the wire request and the corrected metric from
the captured raw record/artifact, not only a helper result.

Because the target-only merge intentionally excludes upstream Python source,
the A/B leg materializes exact commit `324bb05773b3f99743c6516018f3c30cfe33de0b`
as a temporary detached worktree. It hard-fails on a wrong Git identity or an
import outside that worktree, invokes that revision through `python -m aiperf`,
uses disjoint native/Python artifact roots, requires positive mock workload
captures, and removes the worktree after the run. Both products consume one
fixed authored dataset. The mock compares the complete unmodified outbound body
multiset byte-for-byte; issuance order is excluded because dataset scheduler
ordering is not part of #51. Raw SSE projections and corrected ITL remain exact
per-run assertions.

## Sol implementation plan

This is the separately labeled Sol plan for the approved design.

1. Add RED authoring tests for flag-only projection, YAML preservation and
   explicit overlay, typed JSON aliases, default false, all three invalid tuples,
   a valid tuple, and named endpoint-profile validation.
2. Add `per_chunk_usage` to the CLI flags, `Inputs`, typed `Endpoint`, protocol-v2
   endpoint profile DTO, raw/prepared endpoint configs, resolver, YAML overlay,
   and profile registration. Implement one validation helper used by both the
   default endpoint and named profiles.
3. Add RED chat-body tests for option off/on, preservation of unrelated keys,
   explicit false values, and non-object authored `stream_options`. Generalize
   the existing include-usage helper to default both fields without replacement.
4. Add RED reduction/transport tests with role-only, reasoning, output,
   usage-only, final-only, missing-usage, and multiple content frames. Extend
   `ObservedUsage` with an optional first-content cumulative completion count;
   set it exactly once on the first meaningful response only when the endpoint
   opts in.
5. Add RED record and metrics tests porting every applicable upstream case.
   Extend `TokenCounts`, compact observer storage, and `RecordIngest`; add a
   shared checked decode-token divisor and use it in the primary accumulator and
   adaptive record/event paths. Add dependent TPS/user assertions.
6. Extend the Rust mock request model and streaming renderer for cumulative
   per-chunk usage and deterministic first-chunk bundling. Test reasoning and
   plain-output streams at the mock layer.
7. Add a native E2E test that launches the production binary against the mock,
   requests raw/JSON artifacts, and asserts the request option, bundled first
   content count, OSL, ITL, and TPS/user relationship.
8. Run focused tests at every RED/GREEN slice, then formatting, runtime tests with
   and without `engine`, CLI tests, mock-server tests, E2E, clippy over changed
   targets, and diff checks using `sccache` and `/mnt/4tb` target directories.
9. Perform a full Graham review over the exact branch range, repair all Critical
   and Important findings with regressions, rerun the full matrix, and request an
   independent Graham review. Only after approval update tracker and closure
   receipts.

## Source anchors

- `rust/cli/src/flags.rs`, `rust/cli/src/load.rs`, `rust/cli/src/yaml.rs`
- `rust/runtime/src/config/`, `rust/runtime/src/endpoints/`
- `rust/runtime/src/transport/reduce.rs`, `rust/runtime/src/transport/http/sink/`
- `rust/runtime/src/dispatch/sink.rs`, `rust/runtime/src/metrics.rs`
- `rust/runtime/src/metrics_core/`, `rust/runtime/src/adaptive_core/window.rs`
- `rust/mock-server/src/models.rs`, `rust/mock-server/src/handlers.rs`
- `rust/e2e-tests/`
