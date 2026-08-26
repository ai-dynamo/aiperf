# Commit 052 — `810fd8bdd4`

## Upstream intent

Upstream aligns the vLLM per-request speculative-decoding adapter with the
reviewed wire contract from vLLM PR #48915. The statistics move from
`choices[0].speculative_decoding_stats` to response-root
`metrics.speculative_decoding`. In a streaming response they ride the trailing
usage chunk, whose `choices` array is empty; non-streaming responses carry the
same root object in the ordinary response body. vLLM emits the trailing chunk
only when `stream_options.include_usage` is true, so chat and completions now
request it for every effective streaming request, independently of server token
counting. Explicit author values remain authoritative, including an explicit
`include_usage: false`, and a turn-level `stream: false` override prevents
injection.

The vLLM histogram also changes from a JSON object with decimal keys to a dense
integer array. Array index `j` is the accepted-draft count and its value is the
number of verification steps in that bucket. The adapter validates the complete
array before dropping zero-count buckets, and, when `num_spec_tokens` is
present, requires exactly `num_spec_tokens + 1` buckets. Old object histograms,
non-integer entries (including booleans), negative values, impossible lengths,
and inconsistent aggregate or detailed counts degrade to no acceptance record
without failing the request.

The Python preformatted-payload cache increments its manifest because the
outbound body changes even when `use_server_token_count` is false. That cache
version change is implementation-specific, not a separate product behavior.

## Native comparison

The earlier native #13 port provides the engine-neutral record, validation,
metrics, exact/sketch/cellular pooling, artifacts, console output, mock fixture,
and a real-profile E2E. Its wire adapter still implements the pre-review shape:

- `extract_vllm_spec_decode_stats` reads a sole choice's
  `speculative_decoding_stats` and suppresses any other choice count;
- `VllmSpecDecodeStats` accepts `BTreeMap<String, u64>`, so it rejects the new
  dense array and still accepts the obsolete object shape;
- the streamed-chat typed codec retains the obsolete finish-only choice field;
- chat and completions add `include_usage` only when both the endpoint config is
  streaming and `use_server_token_count` is true, and they consult the endpoint
  flag rather than the fully merged payload;
- the opt-in mock emits stats on a choice/finish chunk, and the E2E manually
  authors `stream_options.include_usage`, so it cannot detect the missing
  automatic negotiation; and
- completions has no product integration coverage for the acceptance path.

Native has no persistent Python-style mmap cache of preformatted request bytes.
Its `BodyPlan` is derived from the resolved endpoint configuration for the
current run and retained only in that run's runtime dataset. There is therefore
no native manifest version to bump and no stale cross-run payload entry to
invalidate.

## Required native port

1. Extract only an object-valued `metrics.speculative_decoding` from the
   response root, independent of `choices`, and remove the obsolete typed
   per-choice capture.
2. Decode a dense `Vec<u64>`, validate its optional fixed-width bound, convert
   it to the existing sparse canonical `BTreeMap<u64, u64>`, and retain all
   existing aggregate and detailed-count checks.
3. For chat and completions, decide usage negotiation after endpoint and
   turn extras are merged. Inject `include_usage: true` for an effective
   `stream: true` request whether or not server token counting is selected;
   preserve explicit object members and `false`, treat `null` as absent, and
   leave other authored non-object values unchanged for normal server-side
   validation.
4. Make the Rust mock emit the reviewed root shape for chat and completions,
   streaming and non-streaming. Streaming metrics must appear only in the
   usage chunk and therefore disappear when usage was explicitly disabled.
5. Port the upstream unit behavior and strengthen it with real native profiles
   for both OpenAI endpoints. The E2E must omit manually authored
   `stream_options` and omit `--use-server-token-count`, so successful metric
   projection proves that the production request path negotiated the required
   trailing chunk by itself.
6. Update the public reference/tutorial and mark the earlier #13 wire section
   as superseded while preserving its still-current neutral metrics contract.

## Ancestry constraint

At root's requested checkpoint, the isolated port records exact upstream
`810fd8bdd40a1c35b64d487b3b8487f0a71a0f6b` as the second parent of target-only
merge `9fcd17c62e989fc35dd4358418350d458c52dbc5`. Its first-parent tree is unchanged,
so it imports no upstream Python tree; the reviewed native implementation is a
descendant for root to integrate. No cherry-pick is used.

## Test translation map

| Upstream behavior | Native evidence |
| --- | --- |
| Root metrics on non-streaming chat/completions | Endpoint extraction/parser unit tests and mock serialization tests. |
| Metrics on an empty-choice trailing usage chunk | Real HTTP dispatch test and real-profile E2E. |
| Dense histogram to sparse canonical map | Focused parser worked, zero-step, fully rejected, detailed, and malformed-shape tests. |
| Fixed-width histogram validation | Parser tests for shorter/longer arrays and absent `num_spec_tokens`. |
| Automatic usage request independent of token-count source | Exact materialized chat/completions payload tests plus E2E with no manual extra and no server-token-count flag. |
| Explicit opt-out, `null`, merged stream override, and no config mutation | Exact endpoint body tests using endpoint and turn extras. |
| Metrics absent or malformed | Parser/dispatch tests prove clean degradation and the existing no-stats real profile remains clean. |
| Persistent Python mmap manifest | Not applicable: native has no cross-run preformatted-body cache. |

The source-grounded native contract is in
[`../specs/2026-08-25-native-vllm-spec-decode-wire.md`](../specs/2026-08-25-native-vllm-spec-decode-wire.md).

## Implementation and verification receipt

The isolated native implementation is split into reviewable commits:

- `138ab2e2f9`: root extraction, dense normalization, and typed fast-path cleanup;
- `0a0387009b`: post-merge streaming usage negotiation;
- `02f8abceca`: reviewed mock wire and serializer parity;
- `cbc37b999c`: real native product E2E matrix;
- `cf41b9776a` and `6bc724e921`: formatting and review-driven test cleanup; and
- `08c43bef3c`: public documentation and historical-wire supersession.

Focused runtime, formatter, transport, mock-server, serializer, and real-process
E2E suites pass; the E2E result is 15/15. Formatting, diff checks, changed-scope
Clippy, CLI build, and commit hooks pass. The broader runtime+engine suite passes
2,370 tests, with six unrelated failures recorded in the Graham receipt rather
than modified during this port. The complete pass-by-pass review and exact test
receipts are in
`.superpowers/sdd/2026-08-25-native-vllm-spec-decode-wire/graham-review.md`.

Status: complete. Root independently reviewed immutable range
`13fdae44306aa1c04d8ce7a9c71c5c92f53797fe..082457b2de774db8556e9ed481d686ddae79f218`
in two focused Graham passes over parser/dispatch, mock wire, and E2E. The review
approved with no critical, important, or style findings. Campaign row 52 is
closed on that evidence.
