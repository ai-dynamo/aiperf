<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Response parsing and token accounting parity audit

Domain: response ingestion and token accounting — SSE/stream parsing, which chunks
count as tokens, tokenizer behavior, and the ISL/OSL/reasoning-token numbers that
land in the results.

**Python baseline: `origin/main`, git rev `bc359bf8fd`, read at
`/mnt/4tb/aiperf-parity-py-main/src/aiperf/`.** Every `src/aiperf/...` path and
line number below is against that rev. An earlier draft of this report compared
against the local feature-branch checkout, which is 4345 commits ahead with 132
locally-modified Python files; that draft's Python citations were wrong and have
all been re-derived here. See [Withdrawn after baseline
correction](#withdrawn-after-baseline-correction) and [Corrections to the earlier
draft](#corrections-to-the-earlier-draft).

Rust citations are against the working tree at `rust/`. `origin/main` has no
`rust/` tree, so the Rust half of every finding is unaffected by the baseline
correction.

Runtime evidence came from `rust/target/debug/aiperf` against
`rust/target/debug/aiperf-mock-server --fast` and against two purpose-built HTTP
stubs. Each finding states explicitly whether its Rust half is
**runtime-measured** or **code-read**, and whether its Python half is
runtime-measured or code-read. No source file was modified.

Prior backlog: `docs/dev/python-rust-parity-gaps.md` (dated 2026-07-17). Findings
are marked NEW, KNOWN(still-true), KNOWN(partially-fixed), or KNOWN(now-fixed).

## Summary

Severity counts: **2 P0, 5 P1, 4 P2, 1 P3.** Of the 11 findings in the earlier
draft, **9 are still valid, 1 changed scope, and 1 is withdrawn**; the baseline
correction also surfaced **1 new P0** in an endpoint family the branch had
deleted.

The headline risk is unchanged and survives the correction intact: the Rust
client-side output-token count is a count of **content-bearing response events**,
not a tokenization of the generated text. Upstream Python concatenates the
reconstructed output text and encodes it with the tokenizer
(`records/inference_result_parser.py:602-619`), byte-identical to the branch, so
the divergence is real against `origin/main`. Measured: a non-streaming chat
request whose server reported `completion_tokens: 12` is recorded by Rust as
`output_sequence_length = 1.0`. Because ITL divides by the decode-token count and
output throughput / TPOT / goodput all read OSL, every token-derived number in a
non-streaming or multi-token-per-chunk run is wrong by the same factor.

The new P0 is the mirror image, on an endpoint the earlier draft never saw: for
`audio_transcription`, which upstream declares `produces_tokens: false` and
`tokenizes_input: false`, Python's metrics processor suppresses every token metric
by capability flag while Rust publishes them anyway. Measured on Rust:
`input_sequence_length = 550.0` tokens/request and `input_token_throughput =
84,596 tokens/sec` on an audio-only endpoint that never tokenized anything, plus
`output_sequence_length = 0.0` for a real transcript and
`osl_mismatch_diff_pct = -100%` on 4 of 4 requests.

Below those: Rust does not apply the `STREAMING_ONLY` filter, so a non-streaming
run publishes a `time_to_first_token` that is numerically the whole request
latency where Python publishes nothing. A chunk carrying both `reasoning_content`
and `content` is attributed entirely to reasoning in Rust. Input-length
accounting falls back to the dataset's authored count in Rust where Python
reports the metric absent. The request-latency terminal boundary and the
streaming usage merge differ.

Consistent: SSE byte framing and multi-chunk UTF-8, `add_special_tokens`
handling, `use_server_token_count` precedence, usage synonym precedence and
disjoint-cache re-totalization, and — corrected from the earlier draft — the
`per_chunk_usage` / first-content-chunk ITL divisor refinement, which is a
**shared** feature present on both sides with matching validators and matching
degrade-and-warn policy.

## Findings

### 1. Client-side output tokens are a response-event count in Rust, not a tokenization of the text

- **Severity:** P0
- **Status:** KNOWN(still-true) — P0.3
- **Verdict after baseline correction:** STILL VALID. This was risk (a). Both
  Python files in question (`records/inference_result_parser.py`,
  `common/tokenizer.py`) diverge between branch and baseline, but not in the
  regions this finding rests on.
- **Evidence basis:** Python half **code-read** at baseline; Rust half
  **runtime-measured** in both streaming and non-streaming directions.

**Re-derivation of upstream's output-token count.** Upstream computes it in three
steps, none of which counts response events.

First, responses are walked into two text accumulators:

```587:598:src/aiperf/records/inference_result_parser.py
        for response in responses:
            if not response.data:
                continue
            if isinstance(response.data, ReasoningResponseData):
                if response.data.reasoning:
                    reasoning_texts.append(response.data.reasoning)
                if response.data.content:
                    output_texts.append(response.data.content)
            elif isinstance(response.data, ToolCallResponseData):
                output_texts.append(response.data.tool_call_text)
            else:
                output_texts.append(response.data.get_text())
```

Second, each accumulator is joined and encoded, and the token-list length is the
count:

```602:619:src/aiperf/records/inference_result_parser.py
    async def _compute_token_count(
        self, tokenizer: Tokenizer, texts: list[str], separator: str = ""
    ) -> int | None:
        """Compute the number of tokens in the texts by joining them with an optional separator (default none) and encoding with the tokenizer.
        ...
        """
        if not texts:
            return None
        text = separator.join(texts)
        tokens = await asyncio.to_thread(tokenizer.encode, text)
        return len(tokens)
```

Third, the client-side path calls it once per accumulator:

```645:652:src/aiperf/records/inference_result_parser.py
        tokenizer = await self.get_tokenizer(request_record.model_name)
        output_texts, reasoning_texts = self._parse_output_and_reasoning_texts(
            responses
        )
        output_token_count = await self._compute_token_count(tokenizer, output_texts)
        reasoning_token_count = await self._compute_token_count(
            tokenizer, reasoning_texts
        )
```

So upstream's client-side output-token count is
`len(tokenizer.encode("".join(output_texts)))` with an empty separator — one
encode over the whole concatenated generation, independent of how many SSE events
or choices delivered it. The concatenation is deliberate: joining first and
encoding once also captures merges across chunk boundaries that per-chunk
encoding would miss. `common/tokenizer.py`'s branch/baseline divergence is
entirely in vocabulary-enumeration helpers for `--prompt-corpus random`
(`vocab_size`, `all_special_ids`, `valid_token_ids`, `all_token_ids`,
`_tiktoken_internal`); `encode` and its `add_special_tokens=False` pinning are
identical (`common/tokenizer.py:460-461, 477-487, 734-748`), so nothing in the
tokenizer diff touches this count.

**Rust evidence** — one parsed response with non-empty text emits exactly one
classified-token observation:

```92:98:rust/runtime/src/transport/reduce.rs
        } else if !text.is_empty() {
            if !emit.first_token_released.replace(true) {
                (emit.on_first_token)(at_ns.saturating_sub(emit.start_ns));
            }
            emit.obs
                .on_classified_token(emit.uuid, (emit.to_ms)(at_ns), token_kind(data));
        }
```

and the observation is a `+= 1` counter that becomes the visible count verbatim:

```707:719:rust/runtime/src/metrics.rs
    fn on_classified_token(&self, uuid: Uuid, at_ms: f64, kind: ObservedTokenKind) {
        let at_ns = self.relative_ns_from_ms(at_ms);
        if let Some(request) = self.state.borrow_mut().request_mut(uuid) {
            request.token_arrivals_ns.push(at_ns);
            match kind {
                ObservedTokenKind::Output => {
                    request.output_tokens += 1;
                    request.first_output_token_ns.get_or_insert(at_ns);
                }
                ObservedTokenKind::Reasoning => request.reasoning_tokens += 1,
            }
        }
    }
```

```587:593:rust/runtime/src/metrics.rs
                output: if self.use_server_token_count {
                    let reasoning = self.observed_usage.get(3).map(|value| value as u64);
                    completion_tokens
                        .map(|completion| completion.saturating_sub(reasoning.unwrap_or(0)))
                } else {
                    Some(self.output_tokens)
                },
```

No tokenizer is reachable from the dispatch or measurement path:
`grep -rn tokenizer rust/runtime/src/transport/` returns only proxy/body-encoder
matches, and the concatenated `response_text`
(`rust/runtime/src/transport/http/sink/endpoint_dispatch.rs:567,815`) is consumed
only by capture, accuracy, and graph reply construction — never encoded for OSL.

**Runtime confirmation (Rust half).** Same mock server, same tokenizer
(`cl100k_base`), same `--output-tokens-mean 12`, `--endpoint-type chat`:

| run | wire chunks | `output_sequence_length` | `usage_completion_tokens` |
|---|---|---|---|
| non-streaming | 1 | **1.0** | 12.0 |
| `--streaming` | 14 | 12.0 | 12.0 |

The non-streaming body was
`{"choices":[{"message":{"role":"assistant","content":"...12 tokens..."}}],"usage":{"completion_tokens":12}}`.
Streaming happens to agree only because the mock emits exactly one token per SSE
frame.

**Observable user impact.** In the non-streaming case OSL collapses to 1 (or to
the number of choices) regardless of the real output length, so
`output_sequence_length`, `total_osl`, `output_token_count`,
`output_token_throughput`, `e2e_output_token_throughput`, and any OSL-based
goodput SLO are understated by roughly the true OSL. `inter_token_latency`
divides by the decode-token count derived from OSL
(`rust/runtime/src/metrics_core/itl.rs:15-40`,
`rust/runtime/src/metrics_core/accumulator.rs:891-899`; Python
`metrics/types/inter_token_latency_metric.py:60-102`), so it is suppressed
entirely at OSL < 2 and overstated by the same factor otherwise, and
`output_token_throughput_per_user` (`NANOS_PER_SECOND / itl`) is understated. For
streaming servers that pack several tokens per SSE event (TGI, TensorRT-LLM,
batched vLLM `stream_options`) the direction is the same but smaller. Rust does
print a usage-diff panel when client and server counts disagree, but attributes
it to tokenizer mismatch, so the real cause stays hidden.

- **Confidence:** High.

### 2. `audio_transcription` publishes fabricated token metrics in Rust that Python suppresses by endpoint capability

- **Severity:** P0
- **Status:** NEW — surfaced by the baseline correction, risk (e). The branch
  deleted `endpoints/openai_audio_transcription.py` (123 lines upstream), so this
  endpoint family was invisible to the earlier draft.
- **Verdict:** NEW and confirmed.
- **Evidence basis:** Python half **code-read** at baseline; Rust half
  **runtime-measured** against `aiperf-mock-server`.

**Python evidence.** Upstream declares the endpoint's capabilities:

```192:199:src/aiperf/plugin/plugins.yaml
    metadata:
      endpoint_path: /v1/audio/transcriptions
      supports_streaming: false
      produces_tokens: false
      tokenizes_input: false
      supports_audio: true
      requires_form_data: true
      metrics_title: Audio Transcription Metrics
```

and the metrics processor turns each false capability into a disallowed metric
flag before any metric is computed:

```38:48:src/aiperf/post_processors/base_metrics_processor.py
        endpoint_metadata = plugins.get_endpoint_metadata(self.run.cfg.endpoint.type)
        capability_flags = (
            ("produces_tokens", MetricFlags.PRODUCES_TOKENS_ONLY),
            ("tokenizes_input", MetricFlags.TOKENIZES_INPUT_ONLY),
            ("supports_audio", MetricFlags.SUPPORTS_AUDIO_ONLY),
            ("supports_images", MetricFlags.SUPPORTS_IMAGE_ONLY),
            ("supports_videos", MetricFlags.SUPPORTS_VIDEO_ONLY),
            ("produces_videos", MetricFlags.PRODUCES_VIDEO_ONLY),
        )
        for capability, flag in capability_flags:
            if not getattr(endpoint_metadata, capability):
                disallowed_flags |= flag
```

`PRODUCES_TOKENS_ONLY` covers OSL and `total_osl`
(`metrics/types/output_sequence_length_metric.py:29,69`), output token count
(`metrics/types/output_token_count.py:28,64`), reasoning token count
(`metrics/types/reasoning_token_count.py:31,70`),
`total_token_throughput` (`metrics/types/total_token_throughput.py:29`),
`e2e_output_token_throughput` (`metrics/types/e2e_output_throughput_metric.py:32`),
`output_token_throughput` (`metrics/types/output_token_throughput_metrics.py:30`),
the OSL-mismatch family (`metrics/types/osl_mismatch_metrics.py:40,94,159`), and
the usage-diff family (`metrics/types/usage_diff_metrics.py:117,183`).
`TOKENIZES_INPUT_ONLY` covers ISL and `total_isl`
(`metrics/types/input_sequence_length_metric.py:25`), `usage_prompt_tokens`
(`metrics/types/usage_metrics.py:173`), and `prefill_throughput_per_user`.
`PRODUCES_TOKENS_ONLY` is bit 2 of the flag set
(`common/enums/metric_enums.py:756`).

Upstream's transcription response parsing itself is straightforward — the
transcript becomes a `TextResponseData`, so the *record* still carries text:

```85:106:src/aiperf/endpoints/openai_audio_transcription.py
    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        json_obj = response.get_json()
        if json_obj is not None:
            # response_format json / verbose_json: transcript is the "text" field.
            # Use ``is not None`` (not truthiness) so an empty ``{}`` error body
            # yields no transcript rather than falling through to get_text().
            text = json_obj.get("text")
            usage = json_obj.get("usage")
        else:
            # response_format text / srt / vtt: the whole body IS the transcript
            # (not JSON), so fall back to the raw text rather than dropping it.
            text = response.get_text()
            usage = None
        if not text:
            return None
        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=self.make_text_response_data(text),
            usage=usage,
        )
```

The suppression is therefore a deliberate reporting decision at the metrics
layer, not a consequence of the parser producing nothing.

**Rust evidence.** The descriptor carries the same two capability facts:

```208:223:rust/runtime/src/endpoints/tier2.rs
const AUDIO_TRANSCRIPTION_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "audio_transcription",
    aliases: &[],
    description: "OpenAI audio transcription API",
    endpoint_path: Some("/v1/audio/transcriptions"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: false,
    requires_raw_token_ids: false,
    requires_form_data: true,
    requires_polling: false,
    requires_inline_media: true,
    input_modalities: &[Modality::Audio],
    output_modalities: &[Modality::Text],
```

but Rust consumes them only to skip loading a tokenizer, never to filter the
metric catalog:

```1123:1131:rust/runtime/src/engine/online_execution.rs
/// Whether the selected endpoint descriptor requires a real tokenizer.
///
/// AIPerf only tokenizes when the endpoint tokenizes its input or produces
/// output tokens; `base_metrics_processor` filters token metrics by the same
/// two flags, so a descriptor with both false has no token metrics and needs no
/// tokenizer. Descriptor-driven gating avoids enumerating endpoint IDs.
fn endpoint_needs_tokenizer(descriptor: &crate::endpoints::EndpointDescriptor) -> bool {
    descriptor.tokenizes_input || descriptor.produces_tokens
}
```

The comment asserts the equivalence but the filter it names has no Rust
counterpart: `MetricsConfig` carries no endpoint-capability fact at all
(`rust/runtime/src/engine/execute/dataset_build.rs:1117-1146`, quoted in finding
3), and `grep -rn produces_tokens rust/runtime/src/` finds it only in descriptor
declarations, `transport/reduce.rs`, and the two transport sinks. So the tokenizer
is skipped, the ISL/OSL rows are published anyway, and they are published with
values that no tokenizer produced.

**Runtime confirmation (Rust half).** `--endpoint-type audio_transcription
--request-count 4 --audio-length-mean 2 --audio-format wav` against
`aiperf-mock-server --fast`, reading `artifacts/profile_export_aiperf.json`:

| metric | Rust value | Python at baseline |
|---|---|---|
| `input_sequence_length` | **550.0 tokens** (count 4) | suppressed (`TOKENIZES_INPUT_ONLY`) |
| `total_isl` | **2200.0 tokens** | suppressed |
| `input_token_throughput` | **84,596.6 tokens/sec** | no operand (`total_isl` suppressed) |
| `total_token_throughput` | **84,596.6 tokens/sec** | suppressed (`PRODUCES_TOKENS_ONLY`) |
| `output_sequence_length` | **0.0 tokens** (count 4) | suppressed |
| `total_osl` | 0.0 | suppressed |
| `output_token_throughput` | 0.0 | suppressed |
| `e2e_output_token_throughput` | 0.0 | suppressed |
| `osl_mismatch_count` | **4.0 of 4 requests** | suppressed |
| `osl_mismatch_diff_pct` | **-100.0 %** | suppressed |
| `usage_prompt_tokens` | 1.0 | suppressed |
| `audio_duration`, `rtfx` | 2.0 sec, 1054.5 | the endpoint's real metrics |

**Observable user impact.** A transcription benchmark's headline throughput number
is `input_token_throughput = 84,596 tokens/sec` — fabricated. The 550-token ISL is
the dataset composer's authored synthetic input count (finding 6's fallback,
reached here for every request because no tokenizer exists and the multipart body
carries no tokenizable text), presented as a measured prompt length for an
audio-only endpoint. OSL is 0 because `produces_tokens: false` gates the
token-emission branch (`rust/runtime/src/transport/reduce.rs:78`) even though the
transcript text is real and is retained in `response_text`, so the
OSL-mismatch diagnostic fires on 100% of requests at -100% and points the user at
a tokenizer problem that does not exist. `rtfx` and `audio_duration` — the metrics
that actually characterize an ASR run — are buried among a dozen zero and
fabricated token rows. The same mechanism applies to every Rust descriptor with
`produces_tokens: false` (`rust/runtime/src/endpoints/tier2.rs:82,101,120,139`,
`implementation.rs:245,264`, `riva.rs:35,54,75`) — embeddings, rankings, image
generation, Riva ASR/TTS — so this is not confined to transcription.

- **Confidence:** High (Rust runtime-measured; Python suppression read off the
  flag declarations and the processor's capability loop).

### 3. Non-streaming Rust runs publish streaming-only metrics that Python suppresses

- **Severity:** P1
- **Status:** NEW
- **Verdict after baseline correction:** STILL VALID, line-number re-citation only.
  All three Python files involved (`post_processors/base_metrics_processor.py`,
  `metrics/types/stream_latency_metrics.py`, `common/enums/metric_enums.py`) are
  byte-identical between branch and baseline.
- **Evidence basis:** Python half **code-read** at baseline; Rust half
  **runtime-measured**.

**Python evidence** — the metrics processor disallows every `STREAMING_ONLY`
metric when streaming is off, so TTFT/TTST/TTFO/ITL/ICL/decode-duration/TPOT are
not computed at all:

```49:51:src/aiperf/post_processors/base_metrics_processor.py
        if not self.run.cfg.endpoint.streaming:
            disallowed_flags |= MetricFlags.STREAMING_ONLY
```

```24:27:src/aiperf/metrics/types/ttft_metric.py
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS
    )
```

`STREAMING_TOKENS_ONLY` is the `STREAMING_ONLY | PRODUCES_TOKENS_ONLY` union
(`common/enums/metric_enums.py:780`), so declaring it subjects the metric to both
gates. Same flag on `decode_duration`
(`metrics/types/decode_duration_metric.py:22`), `inter_token_latency`
(`metrics/types/inter_token_latency_metric.py:42-45`), `inter_chunk_latency`
(`metrics/types/inter_chunk_latency_metric.py:36`), `time_to_second_token`
(`metrics/types/ttst_metric.py:24`), `time_to_first_output_token`
(`metrics/types/time_to_first_output_token_metric.py:41`), and
`output_token_throughput_per_user`
(`metrics/types/output_token_throughput_metrics.py:59`).

**Rust evidence** — `MetricsConfig` carries no streaming fact, so nothing filters
the catalog by `STREAMING_ONLY`:

```1117:1146:rust/runtime/src/engine/execute/dataset_build.rs
pub(crate) fn metrics_config(
    spec: &MetricsSpec,
    use_server_token_count: bool,
) -> Result<MetricsConfig> {
    ...
    Ok(MetricsConfig {
        slice_duration_ns,
        slos,
        use_server_token_count,
        storage_mode,
        steady_state,
        ..MetricsConfig::default()
    })
}
```

TTFT is then set from the first token arrival unconditionally, and the single
non-streaming response *is* a token arrival (finding 1):

```1583:1589:rust/runtime/src/metrics_core/store.rs
            self.set_nonnegative_i64(
                row,
                MetricTag::TimeToFirstToken,
                record
                    .first_token_ns
                    .map(|timestamp| timestamp - record.start_ns),
            );
```

`grep -rn STREAMING_TOKENS_ONLY rust/runtime/src/` finds the flag only in the
catalog declarations and one comment
(`rust/runtime/src/metrics_core/accumulator.rs:903` asserts "`ttft` is only
populated for streaming records", which the measured run contradicts).

**Runtime confirmation (Rust half).** Non-streaming chat, 5 requests:
`time_to_first_token = 1.4821506 ms`, `request_latency = 1.499202 ms`,
`decode_duration = 0.0409442 ms`. Per-record: `time_to_first_token 1.077469`,
`request_latency 1.096332`.

**Observable user impact.** Every non-streaming Rust run reports a
`time_to_first_token` that is really "request latency minus the response-parse
gap", plus a `decode_duration` that is noise. Python omits both. A user comparing
a non-streaming run across engines sees a TTFT column appear from nowhere with a
value ~2 orders of magnitude larger than a comparable streaming TTFT, and any
TTFT-based SLO/goodput or adaptive-search objective silently binds to that
number. `inter_token_latency`, `inter_chunk_latency`, `time_to_second_token`, and
`time_to_first_output_token` stayed absent in the measured run only because a
single response cannot produce a second arrival — they are not suppressed by
policy. This finding and finding 2 share one root cause: Rust has no
metric-capability filter at all.

- **Confidence:** High.

### 4. A chunk carrying both reasoning and content is attributed entirely to reasoning in Rust

- **Severity:** P1
- **Status:** NEW (P0.5 covers the analogous tool-call+prose case; this is the
  reasoning+prose case and the direction is reversed)
- **Verdict after baseline correction:** STILL VALID. This was risk (c).
  `endpoints/openai_chat.py` lost 81 lines on the branch, but
  `extract_chat_response_data` is byte-identical between branch and baseline — the
  diff is entirely in `format_payload` system-prompt handling and
  `_ensure_include_usage` (request-side). `records/inference_result_parser.py`'s
  `_parse_output_and_reasoning_texts` and
  `metrics/types/time_to_first_output_token_metric.py` are also identical.
- **Evidence basis:** Python half **code-read** at baseline; Rust half
  **code-read**. As flagged in the earlier draft, this finding has no runtime
  backstop on either side — `aiperf-mock-server` emits reasoning or content, never
  both in one chunk.

**Python evidence** — a mixed chunk becomes one `ReasoningResponseData` carrying
both fields:

```327:330:src/aiperf/endpoints/openai_chat.py
        reasoning = data.get("reasoning_content") or data.get("reasoning")

        if reasoning:
            return ReasoningResponseData(content=content, reasoning=reasoning)
```

and the two texts are then split into two accumulators and tokenized separately:

```590:594:src/aiperf/records/inference_result_parser.py
            if isinstance(response.data, ReasoningResponseData):
                if response.data.reasoning:
                    reasoning_texts.append(response.data.reasoning)
                if response.data.content:
                    output_texts.append(response.data.content)
```

TTFO explicitly treats a `ReasoningResponseData` with non-empty `content` as an
output token:

```67:79:src/aiperf/metrics/types/time_to_first_output_token_metric.py
            first_non_reasoning_token_perf_ns: int = next(
                response.perf_ns
                for response in record.content_responses
                if (isinstance(response.data, TextResponseData) and response.data.text)
                or (
                    isinstance(response.data, ReasoningResponseData)
                    and response.data.content
                )
                ...
            )
```

**Rust evidence** — one chunk yields one classification, and any non-empty
`reasoning` wins:

```166:174:rust/runtime/src/transport/reduce.rs
/// Classify a response chunk as reasoning or output for token emission.
pub(crate) fn token_kind(data: &ResponseData) -> ObservedTokenKind {
    match data {
        ResponseData::Reasoning { reasoning, .. } if !reasoning.is_empty() => {
            ObservedTokenKind::Reasoning
        }
        _ => ObservedTokenKind::Output,
    }
}
```

The mixed shape is produced by both parsers:
`rust/runtime/src/endpoints/chat_chunk.rs:105-111` returns
`ResponseData::Reasoning { content, reasoning }` with `content` carried through
unfiltered, against baseline `endpoints/openai_chat.py:330`. Rust does keep the
content in the reconstructed text (`rust/runtime/src/transport/reduce.rs:184-190`),
so only the *classification* differs.

**Observable user impact.** For each mixed chunk, one token moves from
`output_token_count` to `reasoning_token_count`; `output_sequence_length` is
unchanged because it sums both (`rust/runtime/src/metrics_core/ingest.rs:56-59`).
`time_to_first_output_token` and its network-adjusted variant are pushed later, to
the first pure-content chunk, and are suppressed entirely when every
content-bearing chunk also carries reasoning — which is exactly the transition
chunk pattern on DeepSeek-R1 / Qwen3-style `reasoning_content` streams.
`overall_thinking_efficiency` and any reasoning-share reporting shifts up.
`output_token_count` is additionally gated on `> 0`
(`rust/runtime/src/metrics_core/store.rs:1619-1621`), so a stream whose every
chunk is mixed drops the metric rather than reporting a small number.

Runtime-adjacent evidence: on an all-reasoning Qwen3 stream the Rust shape is
`reasoning_token_count 12.0`, `output_token_count ABSENT`,
`time_to_first_output_token ABSENT`, confirming the TTFO/`output_token_count`
suppression mechanics. The attribution split itself is code-verified only.

- **Confidence:** High on the code paths; the mixed-chunk numeric shift is not
  runtime-verified on either side.

### 5. Input sequence length falls back to the authored dataset count in Rust where Python reports it absent

- **Severity:** P1
- **Status:** NEW
- **Verdict after baseline correction:** STILL VALID. This was risk (b).
  `compute_input_token_count`'s bare-text path and its `None` returns are
  byte-identical between branch and baseline; `common/tokenizer.py`'s divergence
  does not reach this path. The finding is now **runtime-measured** as well, via
  finding 2's transcription run.
- **Evidence basis:** Python half **code-read** at baseline; Rust half
  **runtime-measured** (`input_sequence_length = 550.0` on an endpoint that
  tokenized nothing).

**Re-derivation of upstream's ISL.** Upstream's source of truth is
`request_info.payload_bytes` — the exact JSON that went on the wire — decoded once
and walked into tokenizable `texts` plus a chat-shape `messages` view plus any
`pretokenised_token_count`. `turns` are never read. There are three exits:

```400:433:src/aiperf/records/inference_result_parser.py
        pretokenised = inputs.pretokenised_token_count

        tokenizer = None
        # Chat-template path: count role/template wrapping when the user opted in
        # and the payload is chat-shaped. Runs even when ``texts`` is empty (e.g.
        # an image-only message still contributes role/header tokens).
        tokenizer_cfg = self.run.cfg.tokenizer
        if (
            inputs.messages
            and tokenizer_cfg is not None
            and tokenizer_cfg.apply_chat_template
        ):
            tokenizer = await self.get_tokenizer(request_record.model_name)
            templated = await self._compute_chat_template_token_count(
                tokenizer, inputs.messages
            )
            if templated is not None:
                tool_count = await self._compute_tool_texts_token_count(
                    tokenizer, inputs
                )
                return templated + tool_count + pretokenised

        # Bare-text path: join the extracted texts with a space separator.
        if inputs.texts:
            if tokenizer is None:
                tokenizer = await self.get_tokenizer(request_record.model_name)
            text_count = await self._compute_token_count(
                tokenizer, inputs.texts, separator=" "
            )
            if text_count is not None:
                return text_count + pretokenised

        # Pure pre-tokenised input (e.g. token-id embeddings) carries no text.
        return pretokenised if pretokenised > 0 else None
```

So: chat-template render (`add_generation_prompt=True`) plus separately tokenized
tool text plus pretokenised, when the user opted in; otherwise the extracted texts
joined with a **single space** plus pretokenised; otherwise pretokenised alone;
otherwise **`None`**. And when the payload cannot be decoded at all, the function
returns `None` after a warning:

```388:398:src/aiperf/records/inference_result_parser.py
        if inputs is None:
            if not (
                request_record.request_info
                and request_record.request_info.payload_bytes
            ):
                self.warning(
                    "payload_bytes not set on request_info; cannot compute "
                    "input token count"
                )
            return None
```

`None` propagates to `TokenCounts.input`, and the ISL metric raises
`NoMetricValue` on it (`metrics/types/input_sequence_length_metric.py:39-41`), so
the metric is omitted from the report rather than reported as a number. There is
no authored-count fallback anywhere in the upstream path.

**Rust evidence** — both dead ends substitute the authored count:

```171:178:rust/runtime/src/multiturn.rs
        if !extracted.texts.is_empty() {
            return self.add_text_count(extracted.pretokenised_token_count, &extracted.texts);
        }
        if extracted.pretokenised_token_count > 0 {
            return Ok(extracted.pretokenised_token_count);
        }
        Ok(authored_input_tokens)
    }
```

```196:204:rust/runtime/src/multiturn.rs
        let Ok(body) = serde_json::from_slice(body) else {
            return Ok(authored_input_tokens);
        };
        self.count_extracted(
            &endpoint.extract_payload_inputs(&body),
            authored_input_tokens,
        )
```

and the count is a straight passthrough into the metric:

```1618:1618:rust/runtime/src/metrics_core/store.rs
            self.set_optional_u64(row, MetricTag::InputSequenceLength, record.tokens.input);
```

**Runtime confirmation (Rust half).** The transcription run in finding 2 is
exactly this case: a multipart body with no tokenizable text, no tokenizer loaded,
and `input_sequence_length = 550.0` tokens/request published anyway — the
composer's authored synthetic input count.

**Observable user impact.** For endpoints and inputs whose wire body carries no
tokenizable text — a verbatim `raw_payload` / `inputs_json` body, a multipart
audio or image upload, an image-only chat message without `--apply-chat-template`,
a non-JSON body — Rust reports a numeric `input_sequence_length` equal to whatever
the dataset composer authored (the synthetic target, or 0), while Python omits the
metric. Downstream, `total_isl`, `input_token_throughput`, and the
`usage_prompt_tokens_diff_pct` diagnostic are computed against a number that was
never derived from the request. The direction depends on the authored value: it
reads as an exact ISL when it is a synthetic target, and as ISL 0 when nothing was
authored.

- **Confidence:** High (upgraded from medium-high — the Rust half is now
  runtime-measured).

### 6. Request-latency terminal boundary differs

- **Severity:** P1
- **Status:** KNOWN(still-true) — P1.29
- **Verdict after baseline correction:** STILL VALID, line-number re-citation
  only. `metrics/types/request_latency_metric.py` is not among the diverged files;
  `common/models/record_models.py`'s divergence is confined to the
  `first_content_chunk_completion_tokens` helper and the `first_content_chunk_tokens`
  field, not `content_responses`.
- **Evidence basis:** Python half **code-read** at baseline; Rust half
  **code-read** (magnitude not quantified — see Unverified).

**Python evidence:**

```38:49:src/aiperf/metrics/types/request_latency_metric.py
        request_ts: int = record.start_perf_ns

        # Use content_responses to get last response with actual content
        if not record.content_responses:
            raise NoMetricValue(
                "Request latency requires at least 1 non-empty content response."
            )
        final_response_ts = record.content_responses[-1].perf_ns
```

with `content_responses` filtering to responses whose `data` is set:

```1583:1589:src/aiperf/common/models/record_models.py
    def content_responses(self) -> list[ParsedResponse]:
        """Get only responses with actual content (data is not None or empty).

        This excludes usage-only or [DONE] responses that may appear at the end of streaming responses.
        Useful for timing metrics that should measure content delivery.
        """
        return [response for response in self.responses if response.data]
```

**Rust evidence** — the boundary is the transport's terminal instant, taken after
the stream drains past the trailing usage chunk and `[DONE]`:

```810:813:rust/runtime/src/transport/http/sink/endpoint_dispatch.rs
        let result = HttpDispatchResult {
            start_ns: record.start_ns,
            end_ns: record.end_ns.unwrap_or_else(|| self.clock.now_ns()),
            status: record.status,
```

**Observable user impact.** `request_latency` is inflated on Rust by the interval
between the last content chunk and stream close, and because
`inter_token_latency = (latency - ttft) / decode_tokens` and
`decode_duration = latency - ttft` both read it, ITL and decode duration inherit
the inflation. The mock server's `--fast` mode stamps the trailing frames at the
same instant as the last content frame, so I could not quantify the gap; against a
real server it is one network round of the usage frame plus `[DONE]`.

- **Confidence:** High on the code paths; magnitude not quantified.

### 7. Streaming usage is merged per field in Rust and last-non-empty-chunk in Python, and Rust derives an absent total

- **Severity:** P1
- **Status:** KNOWN(partially-fixed) — P0.4. The synonym-precedence and
  disjoint-cache halves are fixed; the merge and derived-total halves are still
  true.
- **Verdict after baseline correction:** STILL VALID.
  `common/models/usage_models.py` is not among the diverged files, so the
  "now fixed" half is confirmed against baseline unchanged.
  `endpoints/anthropic_messages.py` diverges only in `format_payload`
  system-prompt handling and one log-string reformat; its response parsing and
  usage fold are identical.
- **Evidence basis:** both halves **code-read**.

**Now fixed.** P0.4 states "Python returns the first prompt-token synonym as
reported. Rust re-totalizes Anthropic/Bedrock input tokens with cache-read and
cache-write fields." Upstream Python performs the same re-totalization with the
same gate and the same key set:

```213:229:src/aiperf/common/models/usage_models.py
        for key in self.PROMPT_TOKENS_KEYS:
            if key not in self:
                continue
            value = self[key]
            if (
                key in self.DISJOINT_INPUT_KEYS
                and isinstance(value, int)
                and any(k in self for k in self.DISJOINT_CACHE_KEYS)
            ):
                cache = sum(
                    v
                    for k in self.DISJOINT_CACHE_KEYS
                    if isinstance(v := self.get(k), int)
                )
                return value + cache
            return value
        return None
```

against `rust/runtime/src/endpoints/usage.rs:25-49`. The synonym lists for
prompt/completion/total/cache-read/cache-write/cache-miss/reasoning/tool-use and
the nested-details lookup order match key-for-key
(`common/models/usage_models.py:59-132` vs
`rust/runtime/src/endpoints/usage.rs:26-190`).

**Still true — merge semantics.** Python takes the last chunk that carried any
usage, as one atomic object, explicitly so that input/reasoning/output are
mutually consistent:

```508:519:src/aiperf/records/inference_result_parser.py
        usage = find_last_non_empty_usage(responses)
        if usage is None:
            input_token_count = None
            reasoning_token_count = None
            output_token_count = None
        else:
            input_token_count = usage.prompt_tokens
            reasoning_token_count = usage.reasoning_tokens
            output_token_count = self._server_output_minus_reasoning(
                usage.completion_tokens, reasoning_token_count
            )
```

Rust merges field-by-field, latest present value winning per field:

```241:252:rust/runtime/src/transport/reduce.rs
    observed.prompt_tokens = usage
        .prompt_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_tokens);
    observed.completion_tokens = usage
        .completion_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.completion_tokens);
```

For a provider that splits usage across events (input on the opening event, output
on the closing one) Python's ISL under `use_server_token_count` is absent while
Rust's is the opening value. Anthropic specifically is patched on the Python side
— `endpoints/anthropic_messages.py:602-625` folds keys from earlier usage-bearing
responses into the final object so the last-non-empty-chunk merge always sees a
complete dict — so the remaining exposure is other split-usage dialects.

**Still true — derived total.** Rust synthesizes `usage_total_tokens` when the
server omits it; Python preserves the absence
(`common/models/usage_models.py:266`):

```605:613:rust/runtime/src/metrics.rs
                total_tokens: self
                    .observed_usage
                    .get(2)
                    .map(|value| value as u64)
                    .or_else(|| {
                        prompt_tokens
                            .zip(completion_tokens)
                            .map(|(prompt, completion)| prompt.saturating_add(completion))
                    }),
```

**Observable user impact.** `usage_total_tokens` and `total_usage_total_tokens`
appear in Rust reports for servers that never sent a total; server-count ISL
differs on split-usage dialects other than Anthropic.

- **Confidence:** High.

### 8. Speculative-decoding acceptance: Python suppresses multi-sequence records, Rust keeps the last

- **Severity:** P2
- **Status:** NEW
- **Verdict after baseline correction:** **CHANGED — narrowed.** This was risk
  (d). `spec_decode/vllm_adapter.py` is +11/-47, and the diff reverses part of the
  earlier draft's coverage claim: upstream reads the payload from the **response
  root** (`metrics.speculative_decoding`) with a **dense `list[int]`** histogram,
  zero buckets dropped and length validated against `num_spec_tokens + 1` — which
  is exactly what Rust does. The branch instead read a per-choice
  `speculative_decoding_stats` with string histogram keys. So the earlier draft's
  "Rust reads only vLLM's `metrics.speculative_decoding` shape" remark is a
  **match** with upstream, not a coverage gap. What survives is only the
  multi-sequence guard.
- **Evidence basis:** both halves **code-read**; not runtime-verified (no
  spec-decode fixture in the mock-server path exercised).

**Python evidence** — the record is suppressed unless exactly one response carried
stats:

```354:360:src/aiperf/records/inference_result_parser.py
        with_stats = [r for r in responses if r.spec_decode_stats]
        if len(with_stats) != 1:
            return None
        for _entry, AdapterClass in plugins.iter_all(PluginType.SPEC_DECODE_ADAPTER):
            if AdapterClass.can_adapt(responses):
                return AdapterClass.adapt(responses)
        return None
```

The upstream docstring records the reasoning: an `n > 1` streaming request rides
each sequence's stats on its own finish chunk, and a per-request record cannot
attribute request-level `completion_tokens` to a single sequence, "so a mixed
record is worse than none". `n > 1` non-streaming needs no guard because vLLM
populates `metrics.speculative_decoding` only for single-sequence requests
(`records/inference_result_parser.py:338-343`).

**Rust evidence** — each stats-bearing response overwrites the previous, with no
count guard:

```653:659:rust/runtime/src/transport/http/sink/endpoint_dispatch.rs
            if let Some(value) = &server_response.json {
                if captures_spec_decode
                    && let Some(stats) = extract_vllm_spec_decode_stats(value)
                    && stats.as_object().is_some_and(|object| !object.is_empty())
                {
                    spec_decode_stats = Some(stats.clone());
                }
```

**Consistent (corrected).** The payload location and histogram normalization match
upstream key-for-key: root `metrics.speculative_decoding`
(`rust/runtime/src/endpoints/spec_decode.rs:62-68` vs upstream
`spec_decode/vllm_adapter.py:52-57`), dense `Vec<u64>` inflated to a sparse map
with zero buckets dropped (`spec_decode.rs:104-114` vs `vllm_adapter.py:77-100`),
and the `num_spec_tokens + 1` length check (`spec_decode.rs:92-103` vs
`vllm_adapter.py:88-100`). Rust additionally cross-validates the histogram's
weighted sum against `num_accepted_draft_tokens` and the per-step arrays; upstream
validates the element types and length only. Both directions fail closed.

**Observable user impact.** On an `n > 1` streaming request each sequence carries
its own finish-chunk stats. Python deliberately reports nothing; Rust reports the
last sequence's numbers as the request's, so `mean_acceptance_length`,
`draft_acceptance_rate`, `accepted_draft_tokens`, and `spec_decode_steps` become a
single-sequence sample presented as a request aggregate.

- **Confidence:** High on the divergence; not runtime-verified.

### 9. An empty `reasoning_content` alongside a non-empty `reasoning` is handled differently

- **Severity:** P2
- **Status:** NEW
- **Verdict after baseline correction:** STILL VALID.
  `extract_chat_response_data` is byte-identical between branch and baseline.
- **Evidence basis:** both halves **code-read**.

**Python evidence** — a truthiness `or`, so `""` falls through to the alias:

```326:330:src/aiperf/endpoints/openai_chat.py
        content = data.get("content")
        reasoning = data.get("reasoning_content") or data.get("reasoning")

        if reasoning:
            return ReasoningResponseData(content=content, reasoning=reasoning)
```

**Rust evidence** — `Option::or` short-circuits on `Some("")`, and the filter then
discards it without consulting `reasoning`:

```105:111:rust/runtime/src/endpoints/chat_chunk.rs
        if let Some(reasoning) = delta
            .reasoning_content
            .or(delta.reasoning)
            .filter(|value| !value.is_empty())
        {
            return Some(ResponseData::Reasoning { content, reasoning });
        }
```

**Observable user impact.** For a server that emits both fields with
`reasoning_content: ""` and `reasoning: "<text>"`, Python counts the text as
reasoning tokens and Rust drops the reasoning text entirely, falling through to
the tool-call/content branches — so the chunk is either reclassified as output or
dropped (no token at all if `content` is also empty). This shifts
`reasoning_token_count` down and `output_token_count`/OSL down. Narrow: it needs a
server that sends both keys with one empty.

- **Confidence:** High on the code; no known server emits this shape, so
  real-world exposure is unclear.

### 10. SSE repeated `data:` lines

- **Severity:** P2
- **Status:** KNOWN(still-true) — P1.23
- **Verdict after baseline correction:** STILL VALID.
  `common/models/record_models.py`'s branch/baseline divergence does not touch
  `extract_data_content`; `transports/sse_utils.py` is not among the diverged
  files.
- **Evidence basis:** both halves **code-read**.

**Python evidence** — repeated `data:` fields are joined with a newline per the
SSE spec:

```746:765:src/aiperf/common/models/record_models.py
    def extract_data_content(self) -> str:
        """Extract and combine the data contents from the SSE message.

        Per the SSE spec, multiple data fields are combined and delimited by a
        single newline.
        """
        if len(self.packets) == 1 and self.packets[0].name == _SSE_DATA_FIELD_NAME:
            return self.packets[0].value or ""

        return "\n".join(
            packet.value
            for packet in self.packets
            if packet.name == _SSE_DATA_FIELD_NAME and packet.value
        )
```

**Rust evidence** — only the first `data:` field is read:

```124:130:rust/runtime/src/transport/core/sse.rs
    pub fn data(&self) -> Option<&str> {
        self.packets
            .iter()
            .find(|p| p.name == SseFieldName::Data)
            .and_then(|p| p.value.as_deref())
    }
```

**Observable user impact.** A server that splits one JSON chunk across multiple
`data:` lines in a single event yields, in Rust, a truncated first fragment that
fails JSON decode and is silently skipped — so those tokens do not count at all.
Python reassembles and counts them. No mainstream OpenAI-compatible server does
this today, which is why it stays P2.

- **Confidence:** High on the code paths.

### 11. The Python raw endpoint counts a non-JSON body (including `[DONE]`) as output text

- **Severity:** P2
- **Status:** NEW (the `raw`/template contract gap P1.26 is adjacent but is about
  request-side portability)
- **Verdict after baseline correction:** STILL VALID.
  `endpoints/response_mixin.py` is not among the diverged files, and
  `record_models.py`'s `SSEMessage.get_json` / `get_text` are unchanged.
- **Evidence basis:** both halves **code-read**; not runtime-verified for the
  `raw` endpoint.

**Python evidence** — the JMESPath mixin used by `RawEndpoint` falls back to the
raw text body when JSON parsing fails:

```90:96:src/aiperf/endpoints/response_mixin.py
        json_obj = response.get_json()
        if not json_obj:
            if text := response.get_text():
                return ParsedResponse(
                    perf_ns=response.perf_ns, data=self.make_text_response_data(text)
                )
            return None
```

`SSEMessage.get_json()` returns `None` for `"[DONE]"`
(`common/models/record_models.py:773-786`) while `SSEMessage.get_text()` returns
the literal `"[DONE]"` (`common/models/record_models.py:767-771`), so on a
streaming `raw` run the sentinel becomes a `TextResponseData`.

**Rust evidence** — the sentinel is dropped before any endpoint sees it, and a
non-JSON body simply produces no parsed data:

```640:650:rust/runtime/src/transport/http/transport/endpoint_binding.rs
pub fn decode_sse_response(message: &SseMessage) -> Option<ServerResponse> {
    if message.is_done() {
        return None;
    }
    let raw = message.data()?.to_string();
    Some(ServerResponse {
        perf_ns: message.perf_ns,
        json: serde_json::from_str(&raw).ok(),
        raw: Some(raw),
    })
}
```

**Observable user impact.** With `--endpoint-type raw` and streaming, Python adds
one extra "token" for `[DONE]` and includes its arrival in TTFT/ICL/request-latency
candidates, and it counts a plain-text or HTML body as output text; Rust does
neither. Python OSL is inflated by the `[DONE]` tokenization; the boundary metrics
extend to the sentinel's timestamp. Direction is Python-inflates, so it is a
comparison hazard rather than a Rust regression.

- **Confidence:** High on the code.

### 12. An HTTP 200 with no content-bearing chunk produces a less specific diagnostic in Rust

- **Severity:** P3 (residual of a withdrawn P1 — see
  [Withdrawn after baseline correction](#withdrawn-after-baseline-correction))
- **Status:** NEW, diagnostic quality only
- **Evidence basis:** Python half **code-read** at baseline; Rust half
  **runtime-measured** at both 100% and partial rate.

Both engines classify a 200 with empty content as a failed/error record, so the
numbers agree. What differs is the message. Upstream names the actual cause:

```1622:1626:src/aiperf/common/models/record_models.py
            err = InvalidInferenceResultError("Invalid inference result")
            if len(self.responses) == 0 or len(self.content_responses) == 0:
                err.add_note(
                    "No responses with actual content were received from the server (only usage/metadata, null/empty data, or [DONE] markers)"
                )
```

Rust's per-record error is `{"type": "NativeRequestError", "message": "request
failed before the native transport produced a record"}` and, at a 100% rate, the
run-level message is "All N inference request(s) failed. No successful responses
were collected — check the server URL, endpoint path, and response format" —
which points at the URL and endpoint path rather than at the empty completions.

- **Confidence:** High.

## Withdrawn after baseline correction

### Finding 3 of the earlier draft — "An HTTP 200 with no content-bearing chunk is a failed request in Rust and a valid record in Python" (was P1)

**Withdrawn. Not a divergence.** This was my mis-citation, not a branch artifact.
The earlier draft cited `RequestRecord.valid` — the transport-level predicate at
baseline `common/models/record_models.py:1143-1154`, which indeed does not require
content. But the property that gates metric aggregation is
`ParsedResponseRecord.valid`, and upstream requires a content response there:

```1596:1614:src/aiperf/common/models/record_models.py
    @cached_property
    def valid(self) -> bool:
        """Check if the response record is valid.

        Checks:
        - Request has no errors
        - Has at least one content response
        - Start time is before the end time
        - Response timestamps are within valid ranges

        Returns:
            bool: True if the record is valid, False otherwise.
        """
        return (
            not self.has_error
            and len(self.content_responses) > 0
            and 0 <= self.start_perf_ns < self.end_perf_ns < sys.maxsize
            and all(0 < response.perf_ns < sys.maxsize for response in self.responses)
        )
```

`create_error_from_invalid` (`record_models.py:1616-1626`) then converts such a
record into an error record, and `post_processors/metric_record_processor.py:69`
routes it through `error_parse_funcs` rather than the valid path — the same
disposition as Rust's error branch (`rust/runtime/src/metrics_core/store.rs:1666-1669`).
This region of `record_models.py` is identical on the branch too; the +2/-35 diff
is confined to the `first_content_chunk_completion_tokens` helper and the
`first_content_chunk_tokens` field.

The runtime measurement stands as a characterization of Rust and is retained
above as finding 12 (P3, diagnostic wording). It no longer describes a divergence.

**Housekeeping — the failed shell command.** The `--request-count 6` run against
`http://127.0.0.1:18932` that produced no `artifacts/profile_export_aiperf.json`
was the *partial-rate* follow-up to this finding (an HTTP/1.1 keep-alive stub that
hung the client), not the load-bearing 100%-rate experiment. I re-ran it with a
`ThreadingHTTPServer` stub on port 18941 that alternates a real completion and
`content: ""` and sends `Connection: close`, confirmed listening with two `curl`
probes before launching the client. Result: exit 0, 6 raw records, of which the 3
empty-content responses carry
`{"type": "NativeRequestError", "message": "request failed before the native
transport produced a record"}` and `request_count = 3.0` with
`request_latency count = 3`. So the partial-rate consequence is now
runtime-measured on the Rust side — but since upstream Python invalidates the same
records, it is parity, which is what withdrew the finding.

## Corrections to the earlier draft

- **"Checked and consistent" — ITL first-content-chunk divisor.** The earlier
  draft called Rust's `first_content_chunk_tokens` refinement "a Rust-only opt-in
  and out of scope". That was a branch artifact: the branch deleted
  `first_content_chunk_completion_tokens` from `record_models.py` and the
  `first_content_chunk_tokens` field from `TokenCounts`. Upstream has both, gated
  on the same `per_chunk_usage` opt-in, with the same degrade-to-`OSL - 1` policy
  and the same warn-once on an inconsistent server value:
  `metrics/types/inter_token_latency_metric.py:72-97` and
  `records/inference_result_parser.py:518-527` against
  `rust/runtime/src/metrics_core/itl.rs:15-40`. The `per_chunk_usage` validators
  match as well — it implies `use_server_token_count`, requires `chat`, and
  requires streaming, on both sides
  (`rust/runtime/src/endpoints/config.rs:404-414`). This is a **shared feature at
  parity**, now listed under Checked and consistent.
- **Finding 8 coverage remark.** Withdrawn, see finding 8's verdict: Rust's
  root-`metrics.speculative_decoding` / dense-histogram reading matches upstream.
- **Finding 3.** Withdrawn, see above.

## Checked and consistent

- **ITL formula, divisor, and the `per_chunk_usage` refinement.** Both compute
  `(request_latency - TTFT) / decode_tokens` and suppress at `OSL < 2`, both use
  OSL (output + reasoning) rather than `output_token_count`, and both subtract the
  server-reported first-content-chunk completion count when `per_chunk_usage` is
  on, degrading to `OSL - 1` with a once-per-process warning when that count is
  non-positive or at least OSL: `metrics/types/inter_token_latency_metric.py:42-102`
  and `records/inference_result_parser.py:518-527` vs
  `rust/runtime/src/metrics_core/itl.rs:15-40` and
  `rust/runtime/src/metrics_core/accumulator.rs:891-899`.
- **Terminal usage chunk and `[DONE]` exclusion from token/ICL series.** Python
  filters to `content_responses` (`common/models/record_models.py:1583-1589`,
  `metrics/types/inter_chunk_latency_metric.py:56-72`); Rust only pushes a token
  arrival for a non-empty-text parsed response
  (`rust/runtime/src/transport/reduce.rs:92-98`) and derives ICL from those
  arrivals only (`rust/runtime/src/metrics_core/store.rs:1623-1642`). A usage-only
  chunk reconciles usage without producing an arrival on both sides.
- **Empty / role-only first delta is not the first token.** Python's chat
  extractor returns `None` for an empty `content`
  (`endpoints/openai_chat.py:357-360`), and Rust filters the same
  (`rust/runtime/src/endpoints/chat_chunk.rs:131-133`) and additionally guards on
  `!text.is_empty()` before releasing TTFT
  (`rust/runtime/src/transport/reduce.rs:92`). A leading-whitespace chunk is
  non-empty and counts as the first token on both sides.
- **Multi-chunk UTF-8.** Both buffer raw bytes until the `\n\n` / `\r\n\r\n`
  delimiter and only then decode, so a code point split across network chunks is
  never decoded twice: `transports/sse_utils.py:120-196` (bytearray buffer,
  `decode("utf-8", errors="replace")`) vs
  `rust/runtime/src/transport/http/sse/reader.rs`.
- **`add_special_tokens` handling.** Python pins `add_special_tokens=False` for
  both `encode` and `__call__`, with per-tokenizer kwarg remapping
  (`common/tokenizer.py:460-461, 477-487, 732-748`); the Rust
  `TextTokenizer::encode` contract is explicitly "without automatically adding
  model special tokens" (`rust/runtime/src/dataset/tokenizer.rs:46-48`). Both keep
  `skip_special_tokens=False`-equivalent decode semantics
  (`common/tokenizer.py:465`).
- **ISL composition when text is present.** Both join the endpoint-extracted texts
  with a single space, add `pretokenised_token_count`, and prefer a chat-template
  render with `add_generation_prompt=true` plus separately tokenized tool text when
  the user opted in: `records/inference_result_parser.py:400-448` vs
  `rust/runtime/src/multiturn.rs:136-178`. Measured ISL matched the synthetic
  target (20) on the Rust side.
- **`use_server_token_count` precedence.** Both bypass the tokenizer entirely, read
  ISL/reasoning/output from server usage, preserve absence when usage is absent,
  and compute visible output as `completion_tokens - reasoning_tokens` clamped at
  0: `records/inference_result_parser.py:491-572` vs
  `rust/runtime/src/metrics.rs:571-598`.
- **Usage synonym precedence and disjoint-cache re-totalization.** See finding 7's
  "now fixed" section.
- **Reasoning field precedence for the ordinary case.** `reasoning_content` before
  `reasoning`, and reasoning before tool calls before content, on both sides
  (`endpoints/openai_chat.py:326-360` vs
  `rust/runtime/src/endpoints/chat_chunk.rs:103-133`). The chat fast path in Rust
  is pinned to the generic extractor by a differential test
  (`rust/runtime/src/endpoints/chat_chunk.rs:79-93`).
- **Mixed prose + tool-call chunks (P0.5) direction.** Upstream's chat extractor
  now returns a `ToolCallResponseData` carrying **both** `tool_call_text` and
  `content` for a mixed chunk (`endpoints/openai_chat.py:352-355`, with the
  docstring citing ~18% of agentic turns), matching Rust
  (`rust/runtime/src/transport/reduce.rs:191-199`). But
  `_parse_output_and_reasoning_texts` still appends only `tool_call_text` and drops
  the `content` field (`records/inference_result_parser.py:595-596`), so the
  undercount P0.5 describes survives *inside* upstream — an upstream bug rather
  than a Rust divergence. Not restated as a finding here.
- **Malformed mid-stream JSON is skipped, not fatal, on both sides.** Python's
  `get_json()` swallows `orjson.JSONDecodeError`
  (`common/models/record_models.py:612-619`); Rust's chat `parse_response` returns
  `Ok(None)` for a non-object body
  (`rust/runtime/src/endpoints/implementation.rs:319-322`), so `parse_failed` is
  not tripped for the chat endpoint. Upstream's chat extractor likewise degrades an
  unrecognized `object` field to `None` rather than raising
  (`endpoints/openai_chat.py:310-314`).
- **Empty-content 200 disposition.** Both engines invalidate the record; see the
  withdrawal note and finding 12.

## Unverified / needs runtime check

- **Finding 4, mixed reasoning+content chunk.** Needs a fixture emitting
  `{"delta":{"content":"a","reasoning_content":"b"}}`. `aiperf-mock-server` emits
  one or the other, never both. Compare `output_token_count`,
  `reasoning_token_count`, and `time_to_first_output_token`.
- **Finding 6 magnitude.** Needs a server with a measurable gap between the last
  content chunk and stream close (a deliberate delay before the usage frame and
  `[DONE]`) to quantify the Rust request-latency and ITL inflation.
- **Finding 8.** No spec-decode fixture; the `n > 1` streaming case needs a server
  emitting per-sequence finish-chunk stats.
- **Finding 2, other `produces_tokens: false` endpoints.** I measured
  `audio_transcription`. Embeddings, rankings, image generation, and the Riva
  ASR/TTS/NLP families carry the same descriptor fact
  (`rust/runtime/src/endpoints/tier2.rs:82,101,120,139`,
  `implementation.rs:245,264`, `riva.rs:35,54,75`) and should show the same
  fabricated token rows, but I confirmed only the one endpoint.
- **`sequence_distribution.py` (risk (f)).** No finding of mine depends on it. It
  is request-side ISL *targeting*, not response-side accounting. It does touch the
  boundary of this domain in one place: upstream's `VLLMRatioConfig` and
  `SGLangRangeRatioDistribution` subtract a tokenizer-derived
  `num_special_tokens_to_add()` from each drawn ISL before it reaches the wire
  (`common/models/sequence_distribution.py:672-677, 712, 756-761, 1069-1083`),
  which shifts the ISL number that lands in results. The branch removed 591 lines
  from this file, so upstream has behavior the earlier draft never saw, and the
  risk runs toward *missing* upstream behavior rather than crediting branch-local
  behavior. Whether Rust implements the equivalent special-token subtraction is a
  request/dataset-side question and belongs to the endpoint-payloads auditor.
- **`parse_failed` reachability per endpoint.** Rust's `parse_failed` flag turns a
  200 into a failure whenever an endpoint's `parse_response` returns `Err`. I
  confirmed chat cannot trip it, but did not enumerate which of
  `rust/runtime/src/endpoints/tier2.rs`, `kserve.rs`, `riva.rs`, and `sagemaker.rs`
  return `Err` on a partially-valid body, nor whether Python's counterpart merely
  skips. Embeddings is the most likely candidate and overlaps P1.24.
- **Non-streaming field extraction beyond chat.** I compared the chat
  `message`-vs-`delta` selection on both sides but did not diff the non-streaming
  extraction for completions (`endpoints/openai_completions.py`, +12/-24 on the
  branch, re-read but not fully differenced against Rust), embeddings, rankings,
  and the Riva/KServe families.
- **Whether Rust's non-streaming TTFT (finding 3) also reaches SLO/goodput and
  adaptive search inputs.** The metric is present in `*_aiperf.json`; I did not
  confirm whether an SLO or search objective naming `time_to_first_token` binds to
  it in a non-streaming run on the Rust side while failing validation on the
  Python side.
