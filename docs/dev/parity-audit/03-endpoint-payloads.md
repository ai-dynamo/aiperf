<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Endpoint request payload parity audit

Domain: the HTTP request AIPerf actually puts on the wire — URL path, headers,
and JSON/multipart body — per endpoint family present in both the Python
implementation and the native Rust re-implementation (`rust/runtime/src/`).

**Python baseline:** `/mnt/4tb/aiperf-parity-py-main/src/aiperf/`, git rev
`bc359bf8fd` (`origin/main`). All Python paths and line numbers below are
relative to that tree. An earlier revision of this report cited an in-tree
feature branch that had locally reverted five of the exact files in this
evidence set; every Python citation here has been re-read against the baseline,
and two findings were withdrawn as a result (see
[Withdrawn after baseline correction](#withdrawn-after-baseline-correction)).

**Rust baseline:** working tree at `d2375d93b6`, `rust/runtime/src/`.

Method note: every wire claim below was checked against source on both sides,
and the Rust side was additionally confirmed empirically by running a freshly
built `rust/target/debug/aiperf profile` against a loopback HTTP server that
records the exact request line, headers, and body. Captured requests are quoted
verbatim. The Python side is code-only: `AIPERF_RUNTIME_ENGINE=python` did not
switch engines in this tree, so Python bodies are derived from `format_payload`
/ `build_headers` by reading.

## Summary

One body divergence changes the JSON for a common configuration: the completions
endpoint always sends `prompt` as a one-element array in Rust where the baseline
deliberately sends a bare string, against an explicit upstream comment that some
gateways reject the list wrapping. Three header/mode divergences follow: Rust's
`User-Agent` is `aiperf-transport-http/0` instead of `aiperf/<version>`, a
user-supplied `Content-Type` survives to the wire on JSON endpoints in Rust where
Python overwrote it, and `--extra-inputs stream:<bool>` is a real streaming-mode
switch in Rust (it moves `Accept` and the response reader) but a body-only edit
in Python.

The audio-transcription family — invisible to the previous pass — is
substantially at parity: same multipart field names and ordering, same file
descriptor, an equivalent MIME table, and no synthesised
`language`/`response_format`/`temperature`/`prompt`. Its one real divergence is
generic to all three form endpoints: a nested object or array supplied through
`--extra-inputs` is written as JSON by Rust and as a Python `repr` by Python.

`stream_options.include_usage` and `continuous_usage_stats`, URL path assembly,
custom-endpoint interaction, query merge, multi-turn history, the Anthropic
Messages body, the embeddings body, the `X-Session-Affinity` header itself, and
the `max_tokens`/`max_completion_tokens` selection are consistent.

Counts: 1 P0, 3 P1, 6 P2, 1 P3. Of the ten findings in the previous revision,
8 survived, 2 were withdrawn, and 3 are new.

## Findings

### 1. Completions `prompt` is always a JSON array in Rust, a bare string in Python

**Severity:** P0
**Status:** NEW — re-verified against baseline; the quoted Python comment is
upstream, not branch-local (this was #2 in the previous revision)

**Python evidence** — `endpoints/openai_completions.py:43-49`

```python
        payload = {
            # A single prompt goes on the wire as a bare string (the canonical
            # OpenAI form); some gateways reject the list[str] wrapping.
            "prompt": prompts[0] if len(prompts) == 1 else prompts,
            "model": turn.model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }
```

The single-prompt branch and its rationale comment are present verbatim on
`bc359bf8fd`. The `prompts` list itself is built the same way on both sides —
one entry per non-empty text content (`endpoints/openai_completions.py:37-39`) —
so the divergence is purely the unwrapping of the length-1 case.

**Rust evidence** — `rust/runtime/src/endpoints/implementation.rs:1031`

```rust
        let mut payload = Map::new();
        payload.insert(
            "prompt".into(),
            Value::Array(prompts.into_iter().map(Value::String).collect()),
        );
```

There is no single-prompt branch.

**Observable user impact.** `--endpoint-type completions` with the default
`batch_size: 1`. Captured Rust request body:

```json
{"prompt":[" brother Why bastard Wherefore base When my"],"model":"m","stream":false,"max_tokens":4}
```

Python for the same invocation:

```json
{"prompt": " brother Why bastard Wherefore base When my", "model": "m", "stream": false, "max_tokens": 4}
```

The upstream comment names the failure mode: gateways that accept only
`prompt: string` now reject every request. Even where both are accepted, servers
that treat a list `prompt` as a batch may account or shard it differently.

**Confidence:** High — Rust body captured on the wire; Python branch read
directly on the baseline.

### 2. `User-Agent` is a different string

**Severity:** P1
**Status:** NEW — re-verified; baseline is identical to the previously cited
branch text (was #4)

**Python evidence** — `transports/base_transports.py:79-84`

```python
        from aiperf import __version__

        self.user_agent: str = f"aiperf/{__version__}"
        self.base_headers: dict[str, str] = {
            "User-Agent": self.user_agent,
        }
```

`base_headers` is the first layer of `build_headers`
(`transports/base_transports.py:123`), so it reaches every request unless the
user overrides it.

**Rust evidence** — the benchmark sink constructs the transport without ever
calling `with_user_agent`, so the internal default survives to production
traffic. `rust/runtime/src/transport/http/sink.rs:298`

```rust
            HttpTransport::new(clock.clone(), config.client).with_raw_capture(config.capture_raw);
```

`rust/runtime/src/transport/http/transport/http_transport.rs:62`

```rust
            user_agent: "aiperf-transport-http/0".to_string(),
```

**Observable user impact.** Captured on the wire for a default chat profile:

```
user-agent: aiperf-transport-http/0
```

Python sends `user-agent: aiperf/<version>`. Any server-side UA allowlist, WAF
rule, per-client rate limit, or log-based attribution keyed on `aiperf/` stops
matching, and the version is no longer visible to the server operator. A
user-supplied `--header User-Agent:...` still wins on both sides (verified: a run
with `--header 'User-Agent:my-agent'` produced `user-agent: my-agent`).

**Confidence:** High — captured on the wire; both defaults read directly.

### 3. `--extra-inputs stream:<bool>` changes streaming mode in Rust, body only in Python

**Severity:** P1
**Status:** NEW — re-verified (was #5)

**Python evidence** — `extra` is merged into the body and nothing else reads it.
`endpoints/openai_chat.py:66-87`

```python
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model_name or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }
        ...
        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)
```

`Accept` and the SSE reader come from `endpoint.streaming` alone:
`transports/aiohttp_transport.py:187-191`

```python
        accept = (
            "text/event-stream"
            if request_info.model_endpoint.endpoint.streaming
            else "application/json"
        )
```

The baseline does read the *merged* payload in one place — the `stream_options`
gate at `endpoints/openai_chat.py:92` — but that only decides whether
`stream_options` is added to the body, never the transport mode.

**Rust evidence** — the merged `stream` literal becomes the *effective* streaming
mode for the dispatch. `rust/runtime/src/dataset/request.rs:917`

```rust
    let requested_streaming = match merged_literal(plan, overrides, "stream") {
        Some(Value::Bool(streaming)) => *streaming,
        ...
        None => effective_streaming(turn, configured_streaming, supports_streaming, overrides)?,
    };
    let streaming = requested_streaming && supports_streaming;
```

**Observable user impact.** A run *without* `--streaming` but with
`--extra-inputs 'stream:true'` captured this on the wire under Rust:

```
accept: text/event-stream
body: {"messages":[...],"model":"override-model","stream":true,"max_completion_tokens":4,"stream_options":{"include_usage":true}}
```

Python for the same invocation sends `Accept: application/json` and parses the
response with the non-streaming reader — so TTFT and inter-token latency are
absent or wrong. (Python *does* add `stream_options` here, because its gate reads
the merged body; the divergence is `Accept` and the reader, not that key.) The
mirror case (`--streaming` plus `--extra-inputs 'stream:false'`) inverts: Rust
switches back to the non-streaming reader, Python keeps advertising
`text/event-stream`. Rust is arguably the more correct behavior, but it is a
different observable result with no notice.

**Confidence:** High — Rust `Accept` and body captured on the wire; Python
`Accept` derivation and `extra` merge read directly on the baseline.

### 4. A user-supplied `Content-Type` survives in Rust and is overwritten in Python (JSON endpoints)

**Severity:** P1
**Status:** NEW — re-verified; the 3 lines the branch dropped from
`base_transports.py` are not in this merge path (was #6)

**Python evidence** — transport headers are merged *after* endpoint and per-turn
headers, so they win. `transports/base_transports.py:134-137`

```python
        headers.update(request_info.endpoint_headers)
        if request_info.turns and request_info.turns[-1].extra_headers:
            headers.update(request_info.turns[-1].extra_headers)
        headers.update(self.get_transport_headers(request_info))
```

`transports/aiohttp_transport.py:193-197`

```python
        content_type = request_info.model_endpoint.endpoint.request_content_type
        if content_type != RequestContentType.MULTIPART_FORM_DATA:
            headers["Content-Type"] = (
                content_type or RequestContentType.APPLICATION_JSON
            )
```

**Rust evidence** — `Accept` is inserted unconditionally (same as Python), but
`Content-Type` only fills a gap.
`rust/runtime/src/transport/http/transport/http_transport.rs:115-126`

```rust
        headers.extend(static_headers.clone());
        headers.insert(
            http::header::ACCEPT,
            HeaderValue::from_static(if streaming {
                "text/event-stream"
            } else {
                "application/json"
            }),
        );
        headers
            .entry(http::header::CONTENT_TYPE)
            .or_insert(HeaderValue::from_static("application/json"));
```

Same asymmetry in the shared helper:
`rust/runtime/src/transport/http/transport/headers.rs:137`.

**Observable user impact.** `--header 'Content-Type:application/x-custom' --header 'Accept:application/x-acc'`,
captured under Rust:

```
accept: text/event-stream
content-type: application/x-custom
```

Python for the same flags sends `content-type: application/json` (the user's
value is discarded) and likewise overrides `Accept`. So `--header Content-Type:...`
is silently ignored on one engine and honoured on the other.

Scope correction from the audio-transcription pass: this applies to JSON
endpoints only. On form endpoints Rust strips every case-insensitive
`Content-Type` spelling and installs its own boundary-bearing value
(`rust/runtime/src/transport/http/transport/endpoint_binding.rs:395-396`), which
matches Python's effect there (Python omits the header so aiohttp supplies the
boundary, `transports/aiohttp_transport.py:194`).

**Confidence:** High — Rust headers captured on the wire; Python merge order read
directly on the baseline.

### 5. An all-empty multimodal turn renders `content: []` in Rust and `content: ""` in Python

**Severity:** P2
**Status:** NEW — re-verified (was #7; baseline line moved from 334 to 330)

**Python evidence** — `endpoints/base_endpoint.py:327-330`

```python
        # An empty part list would serialise as ``content: []``, which servers
        # reject ("message content parts cannot be empty"). Degrade to the
        # empty string, matching the single-text fast path above.
        return parts or ""
```

**Rust evidence** — `rust/runtime/src/endpoints/implementation.rs:1478`

```rust
    Ok(Value::Array(parts))
```

There is no empty-list degradation; the single-text fast path above it
(`rust/runtime/src/endpoints/implementation.rs:1444`) is otherwise identical to
the baseline's (`endpoints/base_endpoint.py:312-319`), including the
`AIPERF_ENDPOINT_FORCE_CONTENT_PARTS` escape hatch
(`common/environment.py:599`).

**Observable user impact.** A turn whose text contents are all empty and which
carries no media — reachable from a file or trace dataset with blank rows, and
from `AIPERF_ENDPOINT_FORCE_CONTENT_PARTS=1` with no parts. Rust:
`{"role":"user","content":[]}`; Python: `{"role":"user","content":""}`. The Python
comment records that servers reject the former.

**Confidence:** Medium-High on the code divergence (both branches read directly);
not exercised on the wire, because a synthetic prompt always produces one
non-empty text.

### 6. `--extra-inputs max_tokens:0` is rejected by Rust and sent by Python

**Severity:** P2
**Status:** NEW — re-verified (was #8)

**Python evidence** — `extra` is merged verbatim with no bounds check
(`endpoints/openai_chat.py:83-87`), and the two OpenAI endpoints disagree with
each other on a zero from the dataset: chat uses an explicit `None` test
(`endpoints/openai_chat.py:75`, `if max_tokens is not None:`) while completions
uses truthiness (`endpoints/openai_completions.py:51`, `if turn.max_tokens:`), so
`0` is emitted by chat and dropped by completions. Both spellings are present on
the baseline.

**Rust evidence** — `rust/runtime/src/dataset/request.rs:1002`

```rust
fn positive_u32(value: &Value, field: &str) -> Result<u32> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            DatasetError::Validation(format!("effective request {field} must be a positive u32"))
        })
}
```

applied to `max_tokens`, `max_completion_tokens`, and `max_output_tokens`
(`rust/runtime/src/dataset/request.rs:912`). Rust's endpoint formatter also uses
`if let Some(max_tokens)` for the dataset value
(`rust/runtime/src/endpoints/implementation.rs:1045`), so a dataset-authored `0`
would be emitted by both chat and completions rather than dropped by completions.

**Observable user impact.** Rust fails the run loudly:

```
ERROR Native AIPerf run failed: invalid dataset: effective request max_tokens must be a positive u32
```

Python accepted the same flag and put `"max_tokens": 0` on the wire. Reported as
low severity per scope (Rust refuses loudly), but the refusal is not documented
as a behavior change.

**Confidence:** High — Rust rejection reproduced; Python merge read directly.

### 7. Embeddings `max_tokens` is diagnosed by Python and silently dropped by Rust

**Severity:** P2
**Status:** NEW — re-verified, unchanged lines (was #9). Distinct from KNOWN
P1.24, which covers embeddings *response* validation

**Python evidence** — `endpoints/openai_embeddings.py:61-62`

```python
        if turn.max_tokens:
            self.error("Max_tokens is provided but is not supported for embeddings.")
```

**Rust evidence** — the embeddings formatter never reads `turn.max_tokens` and
emits no diagnostic. `rust/runtime/src/endpoints/implementation.rs:1080`

```rust
        let turn = &request.turns()[0];
        let mut payload = Map::new();
        payload.insert("model".into(), ...);
        payload.insert(
            "input".into(),
            Value::Array(turn_texts(turn).into_iter().map(Value::String).collect()),
        );
```

**Observable user impact.** Neither engine puts `max_tokens` on an embeddings
request (captured Rust body: `{"model":"m","input":["wind your bloody flag Look back into your"]}`).
Python logs an ERROR line telling the user the flag was ignored; Rust is silent,
so `--output-tokens-mean` looks accepted.

**Confidence:** High.

### 8. Mismatched authored image `uuids` length raises in Python and truncates in Rust

**Severity:** P2
**Status:** NEW — re-verified (was #10; baseline line moved from 111 to 158)

**Python evidence** — `endpoints/openai_chat.py:158`

```python
            for content, uuid in zip(image.contents, uuids, strict=True):
```

`strict=True` raises `ValueError` when the two lists differ in length.

**Rust evidence** — `rust/runtime/src/endpoints/implementation.rs:1511`

```rust
        for (content, uuid) in media.contents.iter().zip(&media.uuids) {
```

`Iterator::zip` stops at the shorter side.

**Observable user impact.** A dataset that authors 3 image contents and 2 cache
UUIDs fails loudly under Python and silently sends 2 `image_url` parts under
Rust, dropping the third image from the benchmarked payload.

**Confidence:** High on the code divergence; requires a UUID-bearing image
dataset to exercise, which was not run.

### 9. An object/array `--extra-inputs` value is JSON in a Rust form part and a Python `repr` in Python's

**Severity:** P2
**Status:** NEW (from the audio-transcription pass; applies to every form
endpoint: `audio_transcription`, `image_edit`, `video_generation`)

**Python evidence** — every non-file form field is stringified with `str()`.
`transports/aiohttp_transport.py:494-495`

```python
            str_value = str(value).lower() if isinstance(value, bool) else str(value)
            form_data.add_field(key, str_value)
```

`--extra-inputs` values are parsed as JSON before reaching this point, so a list
arrives as a Python `list` and `str()` renders Python's `repr`.

**Rust evidence** — `rust/runtime/src/transport/http/transport/body.rs:131-140`

```rust
fn form_value(value: &Value) -> Result<String, ErrorDetails> {
    match value {
        Value::Bool(value) => Ok(value.to_string()),
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value)
            .map_err(|error| ErrorDetails::other(format!("serialize multipart field: {error}"))),
        Value::Null => Ok(String::new()),
    }
}
```

**Observable user impact.** `--endpoint-type audio_transcription --extra-inputs
'{"timestamp_granularities": ["word"]}'` writes a part body of `["word"]` under
Rust and `['word']` under Python. Single-quoted output is not valid JSON, so a
Whisper-compatible server that JSON-decodes the field rejects the Python form and
accepts the Rust one; a server that string-compares gets two different values.
Rust is the more correct side, but the wire bytes differ for identical
configuration. Booleans, numbers, strings, and `null` (dropped by both) agree.

**Confidence:** High on both code paths; not exercised on the wire (no ASR
dataset was run).

### 10. The `X-Session-Affinity` kill switch is a different environment variable

**Severity:** P2
**Status:** NEW (replaces withdrawn #3 — the header itself is at parity, only its
off switch is renamed)

**Python evidence** — the setting lives on `Environment.HTTP`, whose prefix is
`AIPERF_HTTP_` (`common/environment.py:747-749`), and is named
`X_SESSION_AFFINITY_FROM_CORRELATION_ID` (`common/environment.py:846-849`), so
the user-facing variable is
`AIPERF_HTTP_X_SESSION_AFFINITY_FROM_CORRELATION_ID`. It is read at
`transports/base_transports.py:150-152`. Being a pydantic `bool`, it also accepts
`no`, `off`, and `n` as false.

**Rust evidence** — `rust/runtime/src/transport/http/transport/headers.rs:63-66`

```rust
pub fn session_affinity_header_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| !env_flag_disabled("AIPERF_HTTP_X_SESSION_AFFINITY"));
    *ENABLED
}
```

`env_flag_disabled` accepts only `0` and `false`
(`rust/runtime/src/transport/http/transport/headers.rs:50`). The doc comment
above claims it "Mirrors `Environment.HTTP.X_SESSION_AFFINITY` on the Python
side", but no field of that name exists on the baseline.

**Observable user impact.** A user who disables the derived header with
`AIPERF_HTTP_X_SESSION_AFFINITY_FROM_CORRELATION_ID=0` keeps getting
`x-session-affinity` on every Rust request; a user who sets
`AIPERF_HTTP_X_SESSION_AFFINITY=0` under Python keeps getting it there.
`AIPERF_HTTP_X_SESSION_AFFINITY=no` is also honoured by Python and ignored by
Rust. The default-on header itself, its value (the stable correlation ID), and
its case-insensitive strip of caller-supplied spellings all match
(`transports/base_transports.py:150-152` vs
`rust/runtime/src/transport/http/transport/headers.rs:69-105`).

**Confidence:** High — both env reads and both defaults read directly; the Rust
header was captured on the wire.

### 11. Audio transcription: empty base64 accepted by Python, rejected by Rust; reserved-key drop is silent in Rust

**Severity:** P3
**Status:** NEW (from the audio-transcription pass)

**Python evidence** — `endpoints/openai_audio_transcription.py:111-123` splits on
the first comma and does not check that either side is non-empty, so `"wav,"`
yields `b64_data: ""`, which the transport base64-decodes to zero bytes and sends
as an empty file part (`transports/aiohttp_transport.py:479-492`). A reserved key
from `--extra-inputs` is dropped *with a warning*
(`endpoints/openai_audio_transcription.py:64-78`):

```python
            if key in _RESERVED_PAYLOAD_KEYS:
                self.warning(
                    f"--extra-inputs {key!r} is managed by the endpoint and was ignored."
                )
```

**Rust evidence** — `rust/runtime/src/endpoints/tier2.rs:900-910` rejects an
empty format or empty payload:

```rust
    if format.is_empty() || b64.is_empty() {
        return Err(EndpointError::InvalidRequest(
            "audio content must use non-empty <fmt>,<b64> encoding".into(),
        ));
    }
```

and `rust/runtime/src/endpoints/tier2.rs:842-854` drops a reserved `file` key
with no diagnostic at all.

**Observable user impact.** A degenerate ASR row produces an empty upload (and a
server-side error) under Python and a client-side request failure under Rust.
`--extra-inputs 'file:...'` is ignored by both, but only Python tells the user.

**Confidence:** High on the code paths; not exercised on the wire.

## Checked and consistent

- **`stream_options.include_usage`.** Both engines force
  `include_usage: true` on *every* streaming chat and completions request,
  independent of `--use-server-token-count`, by reading the post-merge `stream`
  value rather than `endpoint.streaming`; both preserve an author-supplied
  `include_usage`, both treat an explicit `stream_options: null` as absent, and
  both leave a non-object `stream_options` untouched
  (`endpoints/openai_chat.py:92-104` and `_ensure_include_usage` at
  `endpoints/openai_chat.py:167-194`; `endpoints/openai_completions.py:63-83`
  vs `rust/runtime/src/endpoints/implementation.rs:1614` with call sites at
  `:425` and `:1050`).
- **`continuous_usage_stats` / `--per-chunk-usage`.** Gated identically. The
  baseline passes `continuous=per_chunk_usage`
  (`endpoints/openai_chat.py:102-104`) and relies on a validator that requires
  `--use-server-token-count`, chat, and `--streaming`
  (`config/endpoint.py:552-576`); Rust spells the conjunction out at the
  injection site (`rust/runtime/src/endpoints/implementation.rs:427`) and carries
  the same three-part validator with the same three messages
  (`rust/runtime/src/endpoints/config.rs:397-411`). Injected only on chat, only
  when opted in, on both sides.
- **Audio transcription body and multipart shape.** Same field set and same
  order — `file`, then `model`, then endpoint `extra`, then per-turn `extra_body`
  — with `file` reserved against user override
  (`endpoints/openai_audio_transcription.py:59-78` vs
  `rust/runtime/src/endpoints/tier2.rs:703-711` and `:829`). Rust's
  `serde_json::Map` is built with `preserve_order`
  (`rust/runtime/Cargo.toml:119`), so insertion order reaches the wire as it does
  for a Python dict. Same file descriptor `{b64_data, filename, content_type}`,
  same `audio.<lowercased fmt>` filename, and an equivalent MIME result: the
  baseline's table (`endpoints/openai_audio_transcription.py:14-24`) and Rust's
  `mp3|mpga|mpeg → audio/mpeg`, `m4a|mp4 → audio/mp4`, `audio/{fmt}` fallback
  (`rust/runtime/src/endpoints/tier2.rs:912-922`) agree on every entry, including
  the mp3/mpga/mpeg collapse — `wav`, `flac`, `ogg`, and `webm` land on the same
  string through Rust's fallback. Neither side synthesises `language`,
  `response_format`, `temperature`, or `prompt`; all four are reachable only
  through `--extra-inputs`. A turn with no audio fails the request on both sides
  before any bytes are sent (`endpoints/openai_audio_transcription.py:50-57` vs
  `rust/runtime/src/endpoints/tier2.rs:693-702`).
- **Form-encoding selection.** `multipart/form-data` is auto-selected from the
  endpoint's `requires_form_data` declaration, and an explicit conflicting
  `--request-content-type` is refused, on both sides
  (`plugin/plugins.yaml:192-199` plus the validator at
  `config/endpoint.py:610-637` vs
  `rust/runtime/src/endpoints/tier2.rs:208-225` plus
  `rust/runtime/src/transport/http/transport/endpoint_binding.rs:595-610`).
  `null`-valued fields are skipped and booleans lowercased by both
  (`transports/aiohttp_transport.py:477`, `:494` vs
  `rust/runtime/src/transport/http/transport/body.rs:61-62`, `:133`).
- **URL path assembly and dedup.** `dedup_path_overlap` is logically identical
  (`transports/aiohttp_transport.py:255-276` vs
  `rust/runtime/src/transport/http/transport/url.rs:16`): empty sub-path, base
  already ending in the full sub-path, and `/v1` + `v1/...` collapse the same
  way. Captured Rust paths for `--url .../v1`, `--url .../v1/chat/completions`,
  `--url .../proxy`, and `--custom-endpoint /v2/chat` were
  `/v1/chat/completions`, `/v1/chat/completions`, `/proxy/v1/chat/completions`,
  and `/v2/chat` — exactly what the Python branch computes. `custom_endpoint`
  is checked for `is not None` on both sides so `""` means "append nothing"
  (`transports/aiohttp_transport.py:235`), and both consult a per-descriptor
  `streaming_path` override (`transports/aiohttp_transport.py:240-242` vs
  `rust/runtime/src/endpoints/metadata.rs:44`).
- **Query-parameter merge.** Existing base-URL params are preserved and endpoint
  params override them on both sides
  (`transports/base_transports.py:178` vs
  `rust/runtime/src/transport/http/transport/url.rs:51`).
- **`max_tokens` vs `max_completion_tokens`.** Same field selection keyed on the
  same flag with the same default `false`
  (`endpoints/openai_chat.py:75-81` and `config/endpoint.py:233-236` vs
  `rust/runtime/src/endpoints/implementation.rs:412` and
  `rust/cli/src/load.rs:387`). Captured Rust body used
  `max_completion_tokens` for chat and `max_tokens` for completions, matching
  Python.
- **`extra` / `extra_body` precedence.** Both engines apply endpoint-level
  `extra` first, then per-turn `extra_body`, and both let a colliding key
  overwrite a generated field including `model` and `stream`
  (`endpoints/openai_chat.py:83-87` vs
  `rust/runtime/src/endpoints/implementation.rs:423-424` /
  `rust/runtime/src/endpoints/implementation.rs:1594`). Captured Rust body with
  `--extra-inputs 'model:override-model'` emitted `"model":"override-model"`.
- **`stream` presence.** Both engines always emit `stream` for chat, completions,
  and the OpenAI dialects that carry it, and both omit it entirely on
  non-streaming Anthropic Messages
  (`endpoints/anthropic_messages.py:495-498` vs
  `rust/runtime/src/endpoints/anthropic.rs:232`).
- **Fields absent when unset.** Neither engine emits `temperature`, `seed`, `n`,
  `top_p`, `stop`, `logprobs`, `echo`, `ignore_eos`, or `min_tokens` unless the
  user supplies them through `extra` / `extra_body`; there are no synthesised
  defaults on either side. Captured Rust bodies contained exactly
  `messages`/`prompt`/`input`, `model`, `stream`, and the token cap.
- **Anthropic Messages body.** Same key order (`model`, `messages`,
  `max_tokens`), same `max_tokens` fallback of `1024`, same `system`
  string-vs-array selection and system-text prepend, same `tools` hoist from the
  latest turn declaring them (`endpoints/anthropic_messages.py:488-506` vs
  `rust/runtime/src/endpoints/anthropic.rs:201`). Auth headers agree:
  `x-api-key` plus a defaulted `anthropic-version`
  (`endpoints/anthropic_messages.py:444-453`, `common/endpoint_auth.py:31-37` vs
  `rust/runtime/src/endpoints/anthropic.rs:76`).
- **Embeddings body.** `{"model": ..., "input": [...]}` with `input` always an
  array on both sides (`endpoints/openai_embeddings.py:79-82` vs
  `rust/runtime/src/endpoints/implementation.rs:1080`). Confirmed on the wire.
- **Multi-turn history.** Full prior history is resent with captured assistant
  turns interleaved, in the same role order, on both engines
  (`endpoints/base_endpoint.py:225` vs
  `rust/runtime/src/endpoints/implementation.rs:1269`). Captured Rust
  three-turn session: `[user]`, `[user, assistant, user]`,
  `[user, assistant, user, assistant, user]`. `raw_messages` splices verbatim,
  an explicit `[]` contributes nothing, and `reset_context` clears accumulated
  messages on both sides.
- **Content-part shapes.** `{"type":"text"}` / `{"type":"image_url"}` /
  `{"type":"input_audio"}` / `{"type":"video_url"}` for chat and the
  `input_text` / `input_image` rename for Responses match
  (`endpoints/base_endpoint.py:380-397`, `endpoints/openai_responses.py:92-97`
  vs `rust/runtime/src/endpoints/implementation.rs:1521`), as does the
  single-text fast path that emits a bare string and its
  `AIPERF_ENDPOINT_FORCE_CONTENT_PARTS` escape hatch
  (`endpoints/base_endpoint.py:301-319` vs
  `rust/runtime/src/endpoints/implementation.rs:1444`). The chat-only
  `uuid`-bearing `image_url` part is present on both sides.
- **Auth header.** `Authorization: Bearer <api_key>` for OpenAI-compatible
  dialects, merged over user headers, on both sides
  (`endpoints/base_endpoint.py:44-49` vs
  `rust/runtime/src/endpoints/implementation.rs:63`).
- **Correlation headers.** `X-Request-ID` and `X-Correlation-ID` with the same
  names and the same `--session-header` rename of the correlation header only
  (`transports/base_transports.py:125-132` vs
  `rust/runtime/src/transport/http/transport/http_transport.rs:100`). The
  additive `X-Session-ID`, `X-SMG-Routing-Key`, `X-Dynamo-Session-ID`, and
  `X-Dynamo-Parent-Session-ID` derivations are all opt-in and off by default on
  both sides (`common/environment.py:839-863` vs
  `rust/runtime/src/transport/http/transport/headers.rs:17-33`); the
  default-on `X-Session-Affinity` is present on both, and only its env kill
  switch differs (finding 10).
- **`Accept`.** `text/event-stream` when streaming, `application/json`
  otherwise, overriding any user-supplied value, on both sides.

## Withdrawn after baseline correction

- **`stream_options.include_usage` is unconditional in Rust and gated in Python.**
  Withdrawn. Filed as P0 #1 in the previous revision on the strength of a
  two-condition gate (`streaming and use_server_token_count`) that exists only on
  the in-tree feature branch. The `origin/main` baseline forces
  `include_usage: true` for every streaming run in both chat
  (`endpoints/openai_chat.py:92-104`, `_ensure_include_usage` at `:167-194`,
  `merged.setdefault("include_usage", True)`) and completions
  (`endpoints/openai_completions.py:63-83`), carrying an explicit comment that
  gating it on an unrelated flag would silently drop vLLM's per-request metrics
  from the trailing usage chunk. Rust matches upstream exactly, including the
  post-merge `stream` read, the author-value preservation, the `null`-as-absent
  handling, and the non-object passthrough. The branch's reverted
  `openai_chat.py` is the regression, not Rust. The related
  `continuous_usage_stats` gate — genuinely conditional upstream — was checked
  in the same pass and is also at parity; see Checked and consistent.

- **Rust sends an `X-Session-Affinity` header Python never emits.** Withdrawn.
  Filed as P1 #3 on the strength of a branch state in which
  `Environment.HTTP.X_SESSION_AFFINITY` was declared but never read. On the
  baseline the field is named `X_SESSION_AFFINITY_FROM_CORRELATION_ID`, defaults
  to `True` (`common/environment.py:846-849`), and *is* read — the header is
  inserted last, after a case-insensitive strip of caller-supplied spellings, at
  `transports/base_transports.py:150-152`, exactly like Rust. Both engines
  therefore send `x-session-affinity: <correlation-id>` by default and both
  override a user's own spelling. What survives is narrower: the environment
  variable that turns it off has a different name and a narrower accepted value
  set on each side, refiled as finding 10.

## Unverified / needs runtime check

- **Per-part headers inside a multipart body.** aiohttp's `FormData.add_field`
  wraps a `str` value in a payload that carries `Content-Type: text/plain;
  charset=utf-8` (and may add `Content-Length` for the byte-valued file part),
  whereas Rust writes a bare `Content-Disposition` for text parts and
  `Content-Disposition` + `Content-Type` for the file part
  (`rust/runtime/src/transport/http/transport/body.rs:65-90`). This is a
  plausible byte-level difference in the form body that cannot be settled by
  reading AIPerf source alone; it needs one captured Python multipart request.
  Benign for spec-compliant parsers.
- **Completions warmup prompt prefix.**
  `rust/runtime/src/endpoints/implementation.rs:1025` prepends
  `WARMUP_SYSTEM_MESSAGE_PREFIX` ("You are in warmup mode. …") to every prompt
  when `request.credit_phase() == CreditPhase::Warmup`, and there is no
  counterpart anywhere in the baseline's `endpoints/`. A run with
  `--endpoint-type completions --warmup-request-count 1 --request-count 1`
  produced two requests whose bodies were byte-identical and carried **no**
  prefix, so this formatter is apparently not the one the warmup dispatch uses.
  Needs a check of which body plan the warmup credit path selects before this can
  be called either a divergence or dead code.
- **`--streaming` on a non-streaming endpoint.** Rust silently clears the flag
  (`rust/runtime/src/endpoints/config.rs:394-396`, `if self.streaming
  && !supports_streaming { self.streaming = false; }`), which changes `Accept`
  and drops `stream` from the body. Whether the baseline refuses, warns, or
  likewise downgrades was not established; `plugin/plugins.yaml:194` declares
  `supports_streaming: false` for `audio_transcription` but the consumer of that
  flag was not traced.
- **Responses, rankings (NIM / Cohere / HF TEI), image generation, image edit,
  video generation, HuggingFace generate, SolidoRAG, raw, and template
  endpoints.** Not exercised on the wire and not read line-by-line in this pass.
  The shared-path findings above (headers, form-field serialization,
  `extra` merge) apply to any of these that route through the HTTP sink, but the
  per-dialect body fields need their own comparison.
- **gRPC (KServe OIP, Riva).** Out of scope for a parity finding: the pre-existing
  backlog records these as native-only
  (`docs/dev/python-rust-parity-gaps.md:1222` "gRPC is native-only",
  `docs/dev/python-rust-parity-gaps.md:1228` "KServe and Riva endpoint families
  are native-only"), and the baseline's `endpoints/` has no counterpart.
  Confirmed: no gRPC transport exists under the baseline's `transports/`.
- **Python-side wire capture.** `AIPERF_RUNTIME_ENGINE=python` with
  `PYTHONPATH=src` did not switch engines in this working tree (the captured
  `user-agent` stayed `aiperf-transport-http/0`), so no Python request was
  observed on the wire. Every Python claim here rests on reading `format_payload`
  and `build_headers`. A working cross-engine capture harness would let findings
  1 through 4 and 9 be asserted byte-for-byte instead.
