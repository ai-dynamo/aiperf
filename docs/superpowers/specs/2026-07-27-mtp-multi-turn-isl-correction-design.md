# Design: MTP Multi-Turn ISL Correction

**Issue:** AIP-1013
**Date:** 2026-07-27

## Problem

With MTP (Multi-Token Prediction) enabled and `rejection_sample_method: synthetic`, vLLM generates
tokens non-autoregressively. The generated token IDs decode to text, and when that text is
re-tokenized (either by AIPerf's client tokenizer or by vLLM itself on the next turn), BPE produces
a different — smaller — token count. Example: vLLM reports `completion_tokens=1024`; AIPerf
re-encodes the streamed text to 949.

This is an inherent MTP tokenization artifact, not a tokenizer mismatch. Setting `--tokenizer` to
the exact model tokenizer does not resolve it.

### What `--use-server-token-count` already fixes

For the current turn's OSL, the flag reads `completion_tokens` from the server's `usage` field
instead of re-encoding, correctly recording 1024. This is correct.

### What it does not fix

On turn N+1, AIPerf sends the 949-token text on the wire as conversation history. The server
processes 949 tokens of input, reports `prompt_tokens` accordingly, and AIPerf records that value
as ISL — also 949 for that history contribution. The delta (75 tokens) is lost.

This causes total TPS to be undercounted across multi-turn runs: OSL credits 1024 output tokens for
turn N, but ISL for turn N+1 only credits 949 for that history contribution, creating an accounting
gap of 75 tokens per turn.

**Scope:** Only `use_server_token_count=True` + MTP + `rejection_sample_method=synthetic` +
multi-turn. All other configurations are unaffected.

## Root Cause

```
Turn N:  server generates 1024 tokens → decoded to text T
         AIPerf stores text T as assistant Turn in session history

Turn N+1: text T sent on wire → server re-tokenizes T → ~949 prompt tokens
           InferenceResultParser reads usage.prompt_tokens → ISL = 949 for T's contribution
```

The delta (1024 − 949 = 75) exists at OSL time in `InferenceResultParser` but is not persisted
anywhere. By ISL time it is gone.

## Design

### Core mechanic

At OSL time for turn N, `InferenceResultParser` already has:
- `server_completion_tokens = 1024` (from `usage.completion_tokens`)
- the output text (from parsed responses)
- the tokenizer (already loaded)

We compute `re_encoded_count = tokenize(output_text) = 949`, derive `delta = 75`, and accumulate
it in per-session state inside `InferenceResultParser`. At ISL time for turn N+1, we add the
accumulated correction to `server_prompt_tokens`.

Tokenization happens exactly once per assistant output, at OSL processing time — where the
tokenizer is already active. No extra tokenization passes at ISL time.

### Changes required

**`src/aiperf/records/inference_result_parser.py`**

1. Add `_session_isl_corrections: dict[str, int]` instance attribute (keyed by session ID).

2. In `_compute_server_token_counts`, after computing `completion_tokens`, if the record belongs to
   a multi-turn session:
   - Tokenize the output text → `re_encoded_count`
   - `delta = completion_tokens - re_encoded_count`
   - `self._session_isl_corrections[session_id] += delta`

3. When computing ISL from `prompt_tokens`, look up the session correction:
   - `corrected_isl = prompt_tokens + self._session_isl_corrections.get(session_id, 0)`

4. Clear the session entry from `_session_isl_corrections` when a session ends (to avoid unbounded
   growth).

### What does NOT change

- `Turn` model — no new fields
- `Worker` — no changes
- `session_manager.py` — no changes
- Wire format — same text is sent; this is a metrics-only correction

### Things to verify during implementation

- How session ID flows through `RequestRecord` — confirm the field name and that it is populated
  for multi-turn records and absent (or None) for single-turn records.
- Whether `InferenceResultParser` processes records for concurrent sessions interleaved — the
  `dict[session_id → correction]` approach handles this correctly as long as session IDs are unique,
  but this should be confirmed.
- Where in `_compute_server_token_counts` the output text is accessible for tokenization (the
  client-side path already does this via `_compute_client_side_token_counts` — use the same access
  pattern).
- Lifecycle hook for session-end cleanup — confirm which message or record signals session
  completion in `InferenceResultParser`.

## Correctness

The correction is additive and exact:

```
corrected_ISL_N+1 = server_prompt_tokens_N+1
                  + sum(completion_tokens_k − re_encoded_count_k
                        for each assistant turn k in history)
```

For single-turn runs: no assistant turns precede turn 0; `_session_isl_corrections` has no entry;
`corrected_isl = server_prompt_tokens` unchanged. Correct.

For MTP-off or non-synthetic rejection: `completion_tokens` and `re_encoded_count` agree; delta = 0;
no effect. Correct.

For multi-turn MTP+synthetic: delta accumulates each turn. Turn N+1 ISL is corrected by the full
accumulated delta from all prior assistant turns. Correct.

## Testing

- Unit test: `InferenceResultParser` with a mocked multi-turn record where `completion_tokens`
  differs from `re_encoded_count` — assert corrected ISL matches expected value.
- Unit test: single-turn record — assert no correction applied (ISL unchanged).
- Unit test: MTP-off (completion_tokens == re_encoded_count) — assert delta = 0, ISL unchanged.
- Unit test: session cleanup — assert `_session_isl_corrections` entry is removed after session end.
- Existing tests must continue to pass unchanged.
