<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: Endpoints (request-build + response-parse — faithful port)

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built — every tier-1/tier-2 dialect, the KServe + Riva families, the open protocol-v2
`vllm_generate` token-native factory, and their online wire lifecycles. Endpoint identity and
ownership have moved to the runner-owned registry (see §4); this spec is authoritative for formatter,
parser, replay, extraction, transport-lifecycle, and parity behavior.
**Grounding:** line-by-line read of `endpoints/{base_endpoint,openai_chat,openai_responses,
_openai_responses_replay,openai_completions,openai_embeddings,chat_embeddings,response_mixin,
protocols,payload_extraction}.py`, `common/models/{model_endpoint_info,extracted_payload}.py`,
`config/endpoint.py`, `plugin/schema/schemas.py` (`EndpointMetadata`), and the endpoint registry
in `plugin/plugins.yaml`.
**Companion:** `2026-07-10-aiperf-transport-rust-port-design.md` (the wire: URL/headers/SSE/
cancellation — endpoints sit *above* it), `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md`
(the records endpoints produce feed the accumulator), `2026-07-11-aiperf-rust-exporters-overhaul-design.md`
(response → record → report). **This is a faithful PORT** (the parse quirks are earned-in-blood),
in deliberate contrast to the exporters overhaul.

---

## 0. Thesis — the seam between workload and transport

An **endpoint** is the API-dialect adapter between the workload (a `Turn`/`RequestInfo`) and a
canonical decoded payload/response shape. It has exactly two responsibilities plus an
input-accounting side job:

1. **`format_payload`** — build the canonical decoded request from turns (chat messages / responses
   input / completions prompt / embeddings input).
2. **`parse_response`** — parse one canonical decoded server response into a
   `ParsedResponse { perf_ns, data, usage }`; the collector derives TTFT/tokens from these.
3. **`extract_payload_inputs`** — a single pass over the built body that yields the tokenizable text
   + media counts for **input-side ISL** accounting.

The parse logic is a minefield of vendor quirks paid for in wrong-metric bugs (the ~18%
agentic-OSL-undercount fix; the ~64%-of-streaming-turns function-call fix). **Port the behavior
exactly, guard with fixtures.** This is redo-*port*, not redo-*clean* — unlike the exporters.

**Rust home:** the `aiperf_runtime::endpoints` module owns the `Endpoint` trait, every native dialect, the
shared body-build skeleton, the input-ISL extractor, and each adapter's capability descriptor. It
remains transport-neutral. The `aiperf_runtime::transport_http` module owns the object-safe
`HttpEndpointBinding` seam plus URL construction, header composition, body encoding, inline-media
fetch, SSE framing, polling, download, cancellation, and decoding back into the canonical response
shape; the rest of `aiperf` retains endpoint parsing, observation, and scheduled result composition.

The complete tier-2 set is built in `endpoints/tier2.rs` and `tier2/flexible.rs`: NIM/Cohere/Hugging
Face rankings, image generation/edit, video generation, Hugging Face generate, NIM embeddings, image
retrieval, Solido RAG, raw, and template. Multipart JSON/binary encoding, request-local inline-media
fetch deduplication, Clock-paced video polling/download, and post-send cancellation across the entire
poll lifecycle live under `transport_http/transport/`. Per-turn endpoint selection and
response/usage/modality observation are wired in `http/endpoint_dispatch.rs`.
`tests/tier2_endpoints_online.rs` proves all dialect families and all four special lifecycles against
real loopback HTTP, including cancellation anchored to the original submit-body send completion and
native image/video report metrics.

`transport_http/transport/endpoint_binding.rs` owns the object-safe `HttpEndpointBinding`, its
metadata-driven implementation, HTTP request lowering, and HTTP/SSE to `ServerResponse` decoding. The
same endpoint implementation can therefore be paired with a future gRPC or WebSocket binding without
transport-specific endpoint subclasses — the five KServe V2 OIP dialects already bind to the native
Tonic transport (`2026-07-12-aiperf-native-grpc-kserve-v2-design.md`), and endpoint parsing sees the
same canonical JSON shape after either HTTP JSON or gRPC protobuf decoding.

---

## 1. The `Endpoint` trait + the base contract

```rust
pub trait Endpoint {
    fn metadata(&self) -> &EndpointMetadata;                    // capability flags (§4)
    fn format_payload(&self, req: &RequestInfo) -> Result<Value>;   // canonical decoded payload
    fn parse_response(&self, resp: &ServerResponse) -> Option<ParsedResponse>;  // canonical response
    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload;         // input-ISL (§3)
    fn build_assistant_turn(&self, record: &RequestRecord) -> Option<Turn>;     // context replay
}
```

Shared base machinery (default impls, overridable per endpoint):

- **`extract_response_data`** = map `parse_response` over `record.responses`, keep the truthy —
  the batch driver over a whole record's response list.
- **Turn → messages skeleton** (`build_messages`): a turn with `raw_messages` is spliced verbatim
  (even `raw_messages=[]` renders synthetically — intentional, don't silently drop a turn); else
  render a `{role, content}` message. **Single-text fast path**: one text, one content string, no
  media → raw string content (an OpenAI/Dynamo compat hotfix — some servers reject a list-of-parts
  when only one text present); otherwise a parts list.
- **Content-part render hooks** — per-endpoint `type` names (the one place chat vs responses
  differ): chat `{"type":"text"|"image_url"|"input_audio"|"video_url", …}` with nested `image_url:
  {url}`; responses `{"type":"input_text"|"input_image"|"input_audio", …}` with **`input_image` a
  plain string url** and **video rejected** (§2). Audio part **must contain a comma** (data-URI
  split into `{data, format}`, default `wav`).
- **Conversation-level vs per-request field sourcing (the FORK-inheritance rule):** `raw_tools` is
  read via `_latest_turn_attr` (walk from the end → inherited by DAG children that don't redeclare);
  `max_tokens` / `extra_body` / `model` come from **`turns[-1]`** (per-request, children don't
  inherit the parent's limits). Port both.
- **`JMESPathResponseMixin`** (used by raw/template endpoints): JMESPath-first extraction
  with an embeddings→rankings→text auto-detect fallback; malformed field degrades, never crashes
  construction.

---

## 2. The parse scars (port behavior-exact; fixture each)

| Scar | Rule | Source |
|---|---|---|
| **max_tokens switch** | `use_legacy_max_tokens` → wire `max_tokens`; else `max_completion_tokens`; emitted only when set. (Responses uses `max_output_tokens`; completions uses `max_tokens`.) | `openai_chat.py:60-66` |
| **Merge precedence** | base payload < `endpoint.extra` < `turn.extra_body` (extra_body wins — can override model/stream/max_tokens/tools). | `openai_chat.py:68-72` |
| **`_ensure_include_usage`** | force `stream_options.include_usage=true` only when `streaming ∧ use_server_token_count`; preserve an author-set `include_usage` (even `false`). | `openai_chat.py:74-79,129-141` |
| **Chat response precedence** | `reasoning > content+tool_calls > tool_calls > content`. `reasoning = reasoning_content or reasoning` wins outright (carries content). The tool-call branch is the **mixed emit** — includes `content` only when a non-empty str (the ~18% agentic-OSL-undercount fix; keeps client-OSL == server `usage.completion_tokens`). | `openai_chat.py:211-245` |
| **Tool-call reassembly** | streaming deltas keyed by `index`; **missing index → `len(dict)`, NOT 0** (else parallel tool calls collapse into one slot). Modern: `name` **overwritten**, `arguments` **concatenated** (None→`""`). Legacy `function_call`: both concatenated, streaming-legacy always slot 0. | `openai_chat.py:297-403` |
| **object → data-key** | `chat.completion`→`message` (non-stream); `chat.completion.chunk`→`delta` (stream); **unrecognized `object` → `None`, never raise** (error JSON / proxy page / truncated stream degrade to a failure-record). | `openai_chat.py:189-199` |
| **usage-only frame** | a final SSE frame with `data=None` but `usage` present still yields a `ParsedResponse` — that is how server token counts arrive. | `openai_chat.py:159-162` |
| **Responses input shape** | top-level `input` array (+ leading `user_context_message` item), `instructions` = system prompt at **top level** (not in `input`), `max_output_tokens`. | `openai_responses.py:174-224` |
| **Responses video reject** | `_render_video_part` raises at **format time** (`PART_TYPES[VIDEO]` is empty; inheriting the chat default would emit a bad part AND silently under-count ISL). Surface misconfig immediately. | `openai_responses.py:100-112` |
| **Responses SSE event map** | `output_text.delta`→`Text(delta)`; `reasoning_text.delta`→`Reasoning(delta)`; `output_text.done`→`Text(**text** field, not delta)`; **`function_call_arguments.delta`→`ToolCall(delta)`** (~64% of agentic streaming turns have no other data-bearing event — omit it and TTFT never fires + OSL undercounts every tool turn); `response.completed`→usage. All else → `None`. | `openai_responses.py:315-342` |
| **Responses full precedence** | non-stream `output[]` walk, `reasoning > message > function_call`; function_call counts `name`+`arguments` (server counts them, client must too). | `openai_responses.py:364-479` |
| **Responses replay-unsafe filter** | when capturing the parent's `output[]` for FORK replay, drop the 6 item types `{web_search_call, file_search_call, image_generation_call, code_interpreter_call, computer_call, reasoning}` (only valid with the matching tool config / `previous_response_id`; splicing them into a child's `input` 400s). Safe: `message`, `function_call`. | `openai_responses.py:139-148` |
| **Responses dedup-by-id union** | replay capture = union of `response.completed.response.output[]` (canonical order) + `output_item.done.item` events, deduped by **`id > call_id > item_id`** (synthesize `type::hash` if none), **first-writer-wins** (completed merged before done). Bail to base if a failure event `{response.failed/incomplete/error/error}`. | `_openai_responses_replay.py` |
| **Completions** | `prompt` is a **list** (flattened non-empty content strings); `max_tokens` literal; response `choices[0]["text"]` (both `completion`/`text_completion` objects); WARMUP prefix inlined into every prompt; **degrade-to-None** on unrecognized object. | `openai_completions.py` |
| **Embeddings RAISES** | **the one outlier**: a `data` list whose items are not all `object=="embedding"` dicts **raises `ValueError`** (chat/completions degrade-to-None). No stream/max_tokens/`stream_options`; wire field `input` (list); `max_tokens` set → logged error, silently dropped. Soft-None only for missing JSON / empty data / empty embeddings. | `openai_embeddings.py:93-133` |
| **Three malformed-response policies** | completions/chat **degrade-to-None**; **embeddings raises** on present-but-wrong `data`; responses **never raises** in parse. Preserve the asymmetry. | (cross-file) |

`_openai_responses_replay.py` is a **Python line-count artifact** — merge it back into the responses
module in Rust (no per-file cap).

---

## 3. Input-side ISL accounting (`extract_inputs` — the tokenization contract)

A **single pass** over the `orjson.loads`'d body yields the tokenizable text + media counts for
every payload shape. Feeds ISL metrics — a comparability contract the transport spec omits entirely.

- **Two-phase, early-return anti-double-count:** try the items-array walk (`messages`/`input`); a
  top-level `tools` walk **always** runs; **if an items-array matched, return early** (skip the flat
  fallbacks) so embeddings `input:[str]` isn't also swept by the flat `input` handler.
- **`role|type` disambiguation:** an items-array is only accepted if some item is a dict with a
  `role` or `type` key — distinguishes a chat/responses message array from an embeddings
  `input:[str,…]`.
- **The #1 parity risk — tool-schema serialization:** the top-level `tools[]` schema is text the
  server tokenizes into the prefix of *every* request. It collects `name` + `description` + **the
  `parameters` JSON-schema serialized via `orjson.dumps(parameters)`** — the exact JSON string the
  server sees. **The Rust serializer MUST produce byte-identical compact JSON** (insertion-order
  keys, `,`/`:` separators) or ISL drifts on every tool-using workload. Walk BOTH `tool.function`
  (chat shape) and `tool` (responses shape).
- **Replayed `tool_calls` count toward ISL** — an assistant-history turn's `tool_calls[].function.
  {name,arguments}` are re-tokenized by the server on replay; omitting them under-counts agent-history
  ISL by the whole tool-call content. (Responses: `function_call.{name,arguments}` +
  `function_call_output.output`.)
- **Two text ledgers:** `texts` (bare concat-and-encode ISL path, walk order) AND `tool_texts` (the
  chat-template path uses `apply_chat_template(messages)` which drops tools, so it adds `tool_texts`
  on top). Emit both.
- **Pre-tokenized int-list bypass:** `list[int]` / `list[list[int]]` (OpenAI embeddings token-id
  input) → `pretokenised_token_count += Σ len` — a **separate ISL contribution, never re-tokenized**
  (re-tokenizing token IDs is wrong; missing this silently zero-counts).
- **`messages` template view:** the role/content array for `apply_chat_template`; `None` for non-chat
  shapes (they bare-encode `texts`). Media parts are dropped from the template content (counts
  already captured them; templates need string content).

```rust
pub struct ExtractedPayload {
    pub texts: Vec<String>,              // bare-encode ISL
    pub tool_texts: Vec<String>,         // added on top of the chat-template count
    pub image_count: u32, pub audio_count: u32, pub video_count: u32,
    pub pretokenised_token_count: u64,   // int-list bypass, added directly to ISL
    pub messages: Option<Vec<Message>>,  // chat-template view; None for non-chat
}
```

---

## 4. Capability metadata + the endpoint registry

Each adapter carries an **`EndpointMetadata`** capability descriptor. The flags drive **four request
lifecycles + two metric switches** — this is the endpoint layer's real control flow:

| Flag | Gates |
|---|---|
| `tokenizes_input` | whether client-side ISL tokenization runs + input-token metrics exist |
| `produces_tokens` | whether output-token metrics exist |
| `requires_raw_token_ids` | dataset composition + validation must supply an exact `Turn::raw_token_ids` array (token-native factories such as `vllm_generate`); a representation contract, not an endpoint-ID branch |
| `requires_form_data` | body encoded as multipart `FormData` (not JSON); config auto-derives `request_content_type=MULTIPART` at load, rejects a JSON override |
| `requires_polling` | the whole request routes to async submit → poll status → optional download (video) |
| `requires_inline_media` | media URLs downloaded + base64-inlined pre-dispatch (image_retrieval) |
| `streaming_path` | streaming swaps the URL path (`huggingface_generate` `/generate`→`/generate_stream`) |
| `supports_streaming` | config force-disables `streaming` with a warning if false |
| `endpoint_path`, `metrics_title`, `service_kind`, `supports_/produces_{audio,images,videos}` | path append, display, modality acceptance |

**Ownership — identity lives in the runner registry.** There is no closed `EndpointType` enum, no
separate central metadata table, no Python endpoint manifest, and no Python endpoint-semantic
validators. Each Rust adapter owns an **open string ID plus descriptor**; one frozen,
extension-aware runner registry derives capabilities from these descriptors and creates validated
adapter/config bindings, and runner validation shares the same preparation path as execution. Python
treats endpoint IDs as structural strings and delegates all endpoint semantics to the exact selected
runner. The full identity/registry model is
`2026-07-11-aiperf-runner-owned-endpoint-registry-design.md`; the capability flags, formatter,
parser, replay, extraction, and lifecycle behavior described here remain authoritative.

**Registered dialect families** (id → path → key flags):

- **OpenAI-shaped:** `chat` (`/v1/chat/completions`), `completions`, `responses` (`/v1/responses`),
  `embeddings`, `chat_embeddings`, `nim_embeddings`, `cohere_rankings` (`/v2/rerank`),
  `hf_tei_rankings`, `nim_rankings`, `huggingface_generate` (streaming_path), `image_generation`,
  `image_edit` (form-data), `video_generation` (polling+form-data), `image_retrieval` (inline-media,
  `tokenizes_input=false`), `solido_rag`, `raw`/`template` (`null` path, JMESPath/Jinja2).
- **KServe family (PR-664):** `kserve_chat`, `kserve_completions`, `kserve_embeddings`,
  `kserve_v1_predict`, `kserve_v2_infer`, `kserve_v2_embeddings`, `kserve_v2_rankings`,
  `kserve_v2_vlm`, `kserve_v2_images`. Factories preserve selector extras, tensor shapes and
  datatypes, response fallbacks, embedding reshaping, ranking indexes, VLM media, and typed image
  parameters. Open-registry and engine-v2-only; KServe V1 is a dialect identity, not a
  runner-v1 adapter. The five V2 OIP dialects additionally bind to the native Tonic transport (§0).
- **vLLM token-native (PR-1113):** `vllm_generate`, a protocol-v2-only factory for non-streaming
  `POST /inference/v1/generate`. Its descriptor publishes `tokenizes_input=false`,
  `produces_tokens=true`, token input/output modalities, and `requires_raw_token_ids=true`.
  Formatting maps one typed `Turn::raw_token_ids` vector to `token_ids`, preserves validated
  `sampling_params` and remaining extras, applies `max_tokens` as a set-default, fixes `stream=false`,
  and selects the authored/effective model (ports `endpoints/vllm_generate.py:21-142`, moving
  integer-array validation out of the dispatch hot path). The parser accepts `choices[0].token_ids`
  as a non-text `ResponseData::TokenIds`, retains the exact `u32` values, and reconstructs completion
  usage from the array length. HTTP dispatch emits an output-token observation per returned ID,
  records the raw ID vector in normalized model metadata (alongside vLLM's `request_id` and
  first-choice finish reason), and uses the dataset's exact input length as authoritative prompt
  usage. Direct Graph-IR nodes do not yet carry the linear dataset's raw-token handle, so graph
  preparation rejects `requires_raw_token_ids` descriptors up front rather than deferring a missing-ID
  failure to dispatch; static accuracy likewise rejects raw-token-required endpoints (evaluator
  problems carry semantic text), and agentic validation already requires a streaming text dialect.

**Riva family.** `aiperf_runtime::endpoints::riva` adds the native ASR/TTS/NLP adapters that bind to the
Tonic transport; they follow the same open-descriptor, protocol-v2-only registry discipline.

**Config validators** (`EndpointConfig`): streaming auto-disable when unsupported;
`request_content_type` auto-derived from `requires_form_data` (+ reject a conflicting explicit
override); `type=TEMPLATE` auto-set when a template is given; URL boundary validation (reject
whitespace, require scheme+netloc+**hostname** — catches `http://:8000`, http/https only);
`wait_for_model` coherence (interval/mode need a timeout). Defaults: `TIMEOUT = 6h` (vLLM bench
default), `WAIT_FOR_MODEL_TIMEOUT = 0` (probe off), streaming off, `POOLED` reuse. These validators
run inside the runner's shared preparation path, not as a separate Python layer.

The endpoint config carries the wire knobs (`headers`, `api_key`→Bearer, `url_params`, `streaming`,
`use_legacy_max_tokens`, `use_server_token_count`, `extra`, `session_header`, `connection_reuse`,
`download_video_content`) — most consumed by `aiperf_runtime::transport_http`; the body-relevant ones
(`use_legacy_max_tokens`, `use_server_token_count`, `extra`, `primary_model_name`) by the endpoint.

---

## 5. Rust shape + scope

- **In (`aiperf_runtime::endpoints`):** the `Endpoint` trait; tier-1 chat, responses, completions,
  embeddings, and chat-embeddings dialects; every tier-2, KServe, Riva, and `vllm_generate` adapter
  named above; the shared body-build skeleton and content-part hooks; the `extract_inputs` ISL walk
  (§3, including tool-schema byte-parity); each adapter's open-ID `EndpointMetadata` descriptor; and
  config validators. Raw/template use the Rust `jmespath` and `minijinja` implementations, with safe
  template-file resolution.
- **Built wire lifecycles (`aiperf_runtime::transport_http`):** the `HttpEndpointBinding` translation seam,
  multipart encoding, async polling and content download, inline-media retrieval/deduplication,
  endpoint-specific streaming paths, and canonical HTTP/SSE response decoding. All sleeps and
  cancellation deadlines use the injected `Clock`; polling retains one cancellation deadline rooted
  at the original submission's captured send completion.
- **Not here (in `aiperf_runtime::transport_http`):** URL construction (`build_url`/`_dedup_path_overlap` — the
  `/v1`+`v1/…` collapse), header composition (correlation header under `session_header`), SSE framing,
  cancellation. Endpoints build the body + parse decoded JSON only.
- **Testing (parity fixtures):** a Python twin emits, per quirk, `{turns → wire body}` and
  `{response → ParsedResponse}` goldens — the chat precedence + mixed-emit, tool-call reassembly
  (missing-index, modern vs legacy concat), the responses SSE event map + `function_call_arguments`,
  the dedup-by-id union + replay-unsafe filter, embeddings-raises-vs-degrades, and the `extract_inputs`
  walk **including byte-identical tool-schema `parameters` serialization** (the #1 ISL parity gate).

## 6. Resolutions and remaining parity note

1. **Tool-schema JSON byte-parity** — confirm the Rust JSON serializer (serde_json compact) matches
   `orjson.dumps` on key order (insertion order) + separators for the `parameters` schema. If serde
   can't guarantee insertion order, carry the schema as the original bytes from the dataset rather
   than re-serializing. This is the single most fragile ISL parity point.
2. **`Endpoint` object-safety vs monomorphized — resolved.** The object-safe, thread-safe trait is
   held as `Arc<dyn Endpoint>` by the per-turn resolver, allowing one dataset to select different
   dialects without branching on endpoint enums in the dispatcher.
3. **The `raw`/`template` dependency cost — resolved.** The implementation uses `jmespath` and
   `minijinja`; a configured template path must resolve to a canonical regular file without symlink
   path components, while a non-path value remains a literal inline template.
