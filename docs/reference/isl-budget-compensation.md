---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: ISL Budget Compensation Derivation
---

# ISL Budget Compensation Derivation

This page derives the math behind AIPerf's chat-template overhead compensation. If you just want to understand what the system does at a high level, read [Input Sequence Length (ISL) Tokenization](./isl-tokenization.md) first — that page is non-mathematical. This page is for users who want to know why the probe is structured the way it is, or are debugging an unexpectedly high or low ISL on a specific model.

> **Opt-in:** chat-template wrapping compensation (component **(b)** below) only runs when `--apply-chat-template` is set. Without the flag the composer skips the probe and synthetic ISL passes through at its bare-text token count; the marker compensations (**(a)** and **(c)**) still run as documented since they are independent of chat-template behavior.

## What we are trying to compensate

When a user runs `aiperf profile --isl 1000`, the synthetic composer needs to generate a bare prompt (the text inside a single user message's `content` field) of some length `N` such that, after the server applies its chat template and AIPerf injects any cache-bust marker, the wire payload that the model actually processes contains approximately 1000 tokens.

We split the total wire-token cost into three components, each compensated at a different point in the pipeline:

| Component | What it represents | Where AIPerf compensates |
| --- | --- | --- |
| **(a) Cache-bust marker** | Hex-string injected into a message to defeat KV cache reuse. | Reduce either the first user turn's bare prompt OR the synthetic shared system prompt by the marker's token cost, depending on where the worker actually places the marker. |
| **(b) Chat-template wrapping** | Role headers, end-of-turn tokens, BOS, and the assistant-prompt suffix that the model server's tokenizer adds on top of the bare content. | Subtract from every user turn's bare prompt — first turn pays the per-request fixed cost + per-message wrap; later turns pay only the per-message wrap. |
| **(c) System message length when marker lands on system** | When `--cache-bust system_*` lands the marker on the synthetic shared system prompt, the prompt's wire length grows by the marker token cost. | Reduce the synthetic shared system prompt length by the marker cost so the wire system message still matches `--shared-system-prompt-length`. |

Component (a) only ever has a non-zero value when the user actually has cache-bust enabled. `BenchmarkConfig.validate_cache_bust_compatibility` (in `src/aiperf/config/config.py`) refuses `--cache-bust` when a profiling phase explicitly declares a non-`agentic_replay` `timing_mode`, and refuses it outside `--endpoint-type chat` / `responses` — those checks raise as separate `ValueError`s. It does **not** fire when a `scenario` is set (scenario locks apply post-construction) or when no phase has an explicit `timing_mode` yet (effective mode is resolved later). That validator is what lets the composer assume the worker really will inject the marker once a config has passed — incompatible combinations that would silently no-op are refused when knowable at construction time, and the composer would otherwise over-subtract by `marker_tokens`. Configurations that fail the validator never reach the composer, so component (a) compensation can be unconditional once `target != NONE` and the routing in "Marker placement routing" decides which slot it lands on.

This page focuses on **(b)**, which is the most subtle of the three.

## The chat-template wrapping model

For every chat template AIPerf cares about (Llama-3, Qwen, Mistral, DeepSeek, GPT-style), the templated wire-token count for a request decomposes cleanly into:

```
wire_tokens(messages, add_generation_prompt=True)
    =  per_request_fixed
     + Σ_{m in messages} (per_msg_wrap + content_tokens(m))
```

where:

- `per_request_fixed` is the BOS token plus the assistant-prompt suffix (`<|im_start|>assistant\n`, `[/INST]`, etc.). It is charged **once per request** regardless of the number of messages.
- `per_msg_wrap` is the role header plus the end-of-turn marker (`<|im_start|>user\n` and `<|im_end|>\n`, or equivalent). It is charged **once per message**.
- `content_tokens(m)` is `len(tokenizer.encode(m["content"]))` — the bare content tokens, which we already know how to compute via the same tokenizer.

The model assumes per-message wrap is symmetric across roles (user vs. assistant). For mainstream open-source templates this holds within ±1 token; the rare templates that emit materially different wraps per role would need a richer probe.

## The two-equation probe

We don't know `per_request_fixed` and `per_msg_wrap` directly — the chat template is an opaque Jinja string. To recover them, we render the template with two structurally different message arrays for each probe sample `S`:

**Single-message prompt:**
```
single = template([user(S)], add_generation_prompt=True)
```
Substituting into the model:
```
len(single) = per_request_fixed + 1 · per_msg_wrap + 1 · bare(S)
```

**Triple-message prompt:**
```
triple = template([user(S), assistant(S), user(S)], add_generation_prompt=True)
```
Substituting:
```
len(triple) = per_request_fixed + 3 · per_msg_wrap + 3 · bare(S)
```

where `bare(S) = len(tokenizer.encode(S))`.

Subtracting the first from the second:
```
len(triple) − len(single) = 2 · per_msg_wrap + 2 · bare(S)
```

Solving for `per_msg_wrap`:
```
per_msg_wrap ≈ (len(triple) − len(single) − 2 · bare(S)) / 2
```

Then back-substitute to recover the fixed cost:
```
per_request_fixed ≈ len(single) − bare(S) − per_msg_wrap
```

The result is rounded to integers and averaged across multiple probe samples to reduce sensitivity to any one sample's tokenization quirks.

### Why `[user, assistant, user]` instead of `[user, user]`

The simpler shape `[user(S), user(S)]` would also let us solve a 2-equation system with one less message. We don't use it because some chat templates explicitly enforce role alternation and reject two consecutive user turns at template time. The `[user, assistant, user]` shape is the smallest pattern that all mainstream open-source templates accept, and it sidesteps the alternation check entirely.

### Why three samples

Three text samples of varying lengths and topics are tokenized; the per-sample `(per_request_fixed, per_msg_wrap)` pairs are averaged. This averages out:

- Sample-specific tokenization edge cases (a sample that happens to tokenize across a special-character boundary differently from typical text).
- BPE merge variability (rare merges that change the token count by ±1 depending on surrounding context).

A single sample is enough to be approximately correct; three samples is enough to be robust without slowing startup. The probe runs once per benchmark run.

## Defensive return values

The probe returns `(0, 0)` (no compensation) in any of these conditions:

- Tokenizer is `None` or has no underlying HuggingFace tokenizer (e.g. tiktoken `--tokenizer builtin`).
- Underlying tokenizer has no `apply_chat_template` method.
- The model has no chat template configured (`apply_chat_template` raises `ValueError`/`TemplateError`).
- Any sample produces a negative `per_msg_wrap` or `per_request_fixed` (defensive — better to skip compensation entirely than over-correct in a pathological case).

In all of these cases the bare prompt is generated at the user's requested ISL with no compensation, and the record processor falls back to bare-text encoding. The composer never crashes the run because of a probe failure.

## Applying the probe results

Once `(per_request_fixed, per_msg_wrap)` are known, the composer subtracts:

| Turn | Adjustment subtracted from `--isl` |
| --- | --- |
| First user turn | `per_request_fixed + per_msg_wrap + first_turn_marker_tokens` |
| Subsequent user turns | `per_msg_wrap` |

The first turn pays the per-request fixed cost because that's the turn that "owns" the BOS and generation-prompt tokens — even though those tokens are emitted once per request, they have to be subtracted from one specific turn's bare-prompt budget, and the first turn is the natural choice.

The cache-bust marker is also charged to the first turn (when it lands there), for the same reason: it's a request-level cost that needs to come out of one turn's budget.

Subsequent turns only pay the per-message wrap because they don't own any request-level overhead — the BOS and gen-prompt are already accounted for, and the marker (if any) is on the first turn, not them.

Floor at 1 so prompt generation stays valid for very small `--isl` values: `isl_after = max(1, isl - adjustment)`. The synthetic generator can always produce a one-token prompt; it cannot produce a zero-token prompt.

## Why a per-turn split matters

A simpler model — averaging the chat template overhead across all messages and subtracting the same constant from every turn — would be wrong for multi-turn requests. Suppose `per_request_fixed = 9` and `per_msg_wrap = 5`, and you run a 5-turn conversation. The averaged-per-turn estimate over a 5-turn probe would be `(9 + 5*5) / 5 = 6.8 ≈ 7` tokens per turn. Subtracting 7 from each turn's budget means:

- 5 turns × 7 = 35 total tokens subtracted.
- Actual overhead: `9 + 5*5 = 34` tokens.

Close, but the per-turn count is wrong: the first turn was over-compensated by ~7 tokens, the others were under-compensated by ~2 each. With our model, the first turn is reduced by `9 + 5 = 14` and each later turn by `5`, totaling `14 + 4*5 = 34` — exact, and per-turn-correct.

Per-turn correctness matters because the synthetic generator sizes each turn independently. If we over-compensate the first turn, the model receives a ~993-token first turn instead of ~1000; if we under-compensate later turns, the model receives ~1002-token later turns instead of ~1000. The split keeps every individual turn close to `--isl`.

## What the record processor does with this

The record processor doesn't need the probe results — it computes ISL from scratch by running the wire payload through `apply_chat_template` directly. The composer's job is to generate text such that the wire payload hits the right token count; the record processor's job is to report what actually went on the wire. The two sides agree because they both delegate to the same chat template, but they don't share intermediate state.

If the probe returns `(0, 0)` (no chat template available), the composer doesn't compensate and the record processor falls back to bare-text encoding. ISL still flows end-to-end, just at the bare-prompt level instead of the templated level.

## Where this is implemented

- **Probe**: `_estimate_chat_template_overheads` in `src/aiperf/dataset/composer/base.py`.
- **Per-turn adjustment math**: `BaseDatasetComposer.first_turn_isl_adjustment` and `subsequent_turn_isl_adjustment` properties, same file.
- **Subtraction at generation time**: `SyntheticDatasetComposer._generate_text_payloads` in `src/aiperf/dataset/composer/synthetic.py`.
- **System-prompt length compensation (component (c))**: `BaseDatasetComposer.__init__` builds a private `model_copy` of the prefix-prompt config with reduced `shared_system_prompt_length` when the marker lands on the system message. The user-facing config is never mutated.

## Component (a): cache-bust marker token cost — design decisions

The marker probe (`estimate_marker_token_cost` in `src/aiperf/timing/strategies/cache_bust.py`) is simpler than the chat-template probe but has its own design choices worth recording.

**8 deterministic samples.** The probe builds 8 distinct markers and averages their token counts. Each marker is generated from a deterministic but distinct `(benchmark_id, recycle_pass, trajectory_index, trace_id)` four-tuple (`("estimator", i, i, f"estimator-{i}")` for `i` in `range(8)`). Decisions:

- **Why 8 (not 4, not 16).** The marker text is `[rid:<12 hex>]` plus orientation-dependent whitespace (`<rid>\n\n` for prefix targets, `\n\n<rid>` for suffix targets) — fixed boilerplate plus a 12-character hex digest. The boilerplate tokenizes identically every time; only the digest varies. Across 8 hex digests we see ~1-token spread for typical BPE tokenizers. 4 samples would also work; 8 hedges against pathological tokenizers that BPE-merge digit runs irregularly. 16 would not improve the rounded result.
- **Why deterministic samples (not random).** A `random.randint`-based probe would produce slightly different rounded compensation across runs of the same benchmark. Wire ISL would then drift by ±1 token between runs, which is small but observable in tight rerun-comparison workflows. Deterministic inputs make the compensation reproducible.
- **Why we don't probe per-conversation.** Each conversation's actual marker is built from the real `(benchmark_id, recycle_pass, trajectory_index, trace_id)` at run time. Per-conversation marker tokenization could give a per-conversation exact compensation, but doing so would require running the tokenizer once per conversation at composition time. The variance in marker token count across runs is sub-token after rounding, so the per-conversation cost isn't worth paying.

**Returns 0 for `CacheBustTarget.NONE`.** Skip the encode round-trip entirely when the user hasn't enabled cache-bust. Tested explicitly.

## Component (c): shared system prompt regeneration — alternatives considered

When the marker lands on the synthetic shared system prompt (i.e., `--cache-bust system_*` and `--shared-system-prompt-length` is set), the wire system message length grows by the marker token cost unless we compensate. We considered four approaches:

| Approach | Used? | Reason |
| --- | --- | --- |
| **`model_copy` the prompt config before passing to `PromptGenerator`** | Yes | Localized, no wasted work, doesn't touch user-facing config. |
| Mutate `config.input.prompt.prefix_prompt.shared_system_prompt_length` in place | No | Other consumers of `BenchmarkConfig` (metrics, exporters, downstream services) would silently read the compensated value; user-typed value would no longer match what code reports. Hidden side effect. |
| Generate the system prompt at user-configured length, then call a public setter to regenerate it shorter | No | Wastes tokenizer work generating then discarding a system prompt. Requires a new public method on `PromptGenerator` whose only caller is this single edge case. Crosses layering boundaries. |
| Add a `shared_system_prompt_length_override` kwarg to `PromptGenerator.__init__` | No | Pollutes a public API with a parameter that is internal to one upstream caller. The `model_copy` approach achieves the same thing without changing `PromptGenerator`'s signature. |

The `model_copy` approach is also the only one that survives a "what if the user later reads the config to log it" review: their typed value `200` is what they see, even though the synthetic prompt was generated at `185`.

**Floor at 1.** When `marker_tokens > configured_length` (pathological: `--shared-system-prompt-length 5 --cache-bust system_prefix`), `max(1, configured - marker) = 1`. The synthetic generator can produce a 1-token prompt; it can't produce a 0- or negative-token one. Tested.

## Marker placement routing — encoded once, mirrors the worker

The composer must decide for itself which slot the worker is going to inject the marker into, because compensation differs by slot. The decision tree:

```
target == NONE                        → no compensation
target ∈ {SYSTEM_PREFIX, SYSTEM_SUFFIX} and shared_system_prompt_length is set
                                      → marker lands on system prompt → component (c)
target ∈ {SYSTEM_PREFIX, SYSTEM_SUFFIX} and shared_system_prompt_length is None
                                      → worker fallback: marker lands on first user turn → component (a)
target ∈ {FIRST_TURN_PREFIX, FIRST_TURN_SUFFIX}
                                      → marker lands on first user turn → component (a)
```

This must agree exactly with `worker._apply_cache_bust` in `src/aiperf/workers/worker.py:257` — if the composer decides "first user turn" but the worker decides "system message", wire ISL drifts by ±`marker_tokens` from the user's `--isl` target. The test suite covers all 9 cells (4 non-NONE targets × {has shared system prompt, has none} + NONE).

The routing also drives whether the marker estimator runs at all. When `target == NONE`, no encode round-trip happens. When `target != NONE`, the estimator runs once and the same token count is reused for whichever slot the routing selected.

## Out of scope — what this compensation deliberately does NOT cover

These are documented here so future maintainers don't try to "fix" them without first understanding why they're left alone.

### Trace-loader synthetic content

`weka_trace`, `mooncake_trace`, `bailian_trace`, `dag_jsonl` produce real trace text. The worker still injects the cache-bust marker into trace `raw_messages`, so wire ISL of trace replays exceeds the trace's natural ISL by ~`marker_tokens` per request. **Why we don't compensate**: trace ISL is data-driven; the user explicitly chose this trace as a workload baseline, and trimming trace text would change the workload semantics. Real-world impact is small (trace ISLs are typically 1k–10k tokens; a 10-token marker is sub-1% drift). Per-loader opt-in trimming would be the right approach if a use case ever requires it; a global compensation is the wrong shape.

### Multi-turn assistant response overhead

In `deltas_without_responses` mode, request K of a K-turn conversation contains the full prior assistant response history. Each prior assistant message contributes `per_msg_wrap + assistant_response_tokens` to wire ISL. **Why we don't compensate**: assistant response tokens are not under AIPerf's control — they're the actual model output at runtime. Compensating per-assistant-turn would require either predicting response length (impossible) or accumulating measured response tokens into subsequent user turn budgets (would make synthetic prompt size depend on prior runtime behavior, breaking reproducibility). Current behavior: wire ISL of request K ≈ K × `--isl` + Σ(actual assistant responses) + small slack.

### Tools and function-call schemas

If a payload includes `tools=[...]`, the server's chat template adds tokens for the tool definitions. AIPerf's client-side estimate doesn't model these. **Why we don't compensate**: tool schemas are user-supplied JSON whose token cost varies wildly. For tool-heavy benchmarks the right answer is `--use-server-token-count`, which is canonical.

### Multimodal content

Image/audio/video content has model-specific token costs (CLIP patches, audio frames, etc.) that a generic chat-template probe can't model. AIPerf already tracks media counts separately. **Why we don't compensate**: any compensation would have to be model-specific and would not generalize across `--tokenizer` choices. Use `--use-server-token-count` for multimodal-inclusive ISL.

The cache-bust marker IS injected into multimodal payloads — when the targeted message's `content` is a list of parts, the worker prepends or appends a `{"type": "text", "text": "<marker>"}` part, mirroring the string-content path. The marker token cost component (a) compensates the marker text exactly the same way it does for text-only payloads; the media token cost is the only thing left uncompensated, and that gap is identical to the gap the chat-template-aware ISL feature has on multimodal in general.

### Tokenizers without `apply_chat_template`

Tiktoken builtin, completions-only models, and custom tokenizer wrappers may not expose `apply_chat_template`. The probe returns `(0, 0)` and no chat-template compensation is applied. **Why this is correct, not degraded**: without a chat template, the wire payload also isn't chat-templated — the request format is plaintext or JSON-as-prompt with no role wrapping. Synthetic content of N tokens really does become N tokens on the wire, so 0 compensation is right.

## Failure modes the design protects against

Each scenario was a real concern during design; each is covered by a defensive code path and a test.

| Scenario | Behavior |
| --- | --- |
| Tokenizer is `None` | All three components are 0; composer behaves as before this feature existed. |
| Underlying tokenizer has no `apply_chat_template` | Component (b) returns `(0, 0)`; (a) and (c) still work. |
| `apply_chat_template` raises (no template configured) | Component (b) returns `(0, 0)`; defensive `try/except` catches all exceptions. |
| Probe returns negative numbers for any sample | Component (b) returns `(0, 0)`; under-compensation is safer than over-compensation. |
| `--isl 5` with 19-token first-turn adjustment | Floor at 1; benchmark still produces a (very short) prompt. |
| `--shared-system-prompt-length 3 --cache-bust system_prefix` with 10-token marker | Floor compensated length at 1; benchmark still produces a system prompt. |
| Cache-bust target is `NONE` | Marker estimator never invoked; encode round-trip skipped. |
| User mutates config after composer init | No effect on compensation — composer reads config once and copies what it needs. |

## Rejected alternatives — full audit trail

Each alternative was considered during design and rejected. Recorded here so the trade-offs aren't re-litigated without context.

1. **Don't compensate at all.** Wire ISL silently exceeds `--isl` by 5–25 tokens depending on benchmark mode, marker setting, and tokenizer. For short prompts (`--isl 50`), this is up to 50% drift. Rejected as silently misleading.

2. **Single-overhead probe with every-turn subtraction.** The first iteration of this feature (and what an earlier draft of `isl-tokenization.md` described). Over-subtracts by `BOS + gen_prompt` per turn after the first in multi-turn benchmarks; the error grows linearly with K. Rejected after a critical review walked the multi-turn flows and identified the over-subtract.

3. **Subtract from first turn only, leave subsequent turns alone.** Single-turn benchmarks would be exactly right; multi-turn would drift by `K × per_msg_wrap` for request K. Rejected because per-turn correctness matters: the synthetic generator sizes each turn independently, so a per-turn drift is more visible than a per-request drift.

4. **Mutate `BenchmarkConfig` in place to compensate the shared system prompt.** Simpler code, but means downstream consumers see different numbers than the user typed. Rejected as hidden side effect.

5. **Add a public `regenerate_shared_system_prompt(length)` setter on `PromptGenerator`.** Would let the composer compensate after the fact. Rejected because it wastes tokenizer work generating the original prompt and crosses a layering boundary; `model_copy` of the prompt config achieves the same thing without those costs.

6. **Add a `shared_system_prompt_length_override` kwarg to `PromptGenerator.__init__`.** Would centralize the compensation in the prompt generator. Rejected because the override is purely a composer-internal concern and shouldn't pollute a public init signature.

7. **Per-role probe distinguishing user and assistant wraps.** Doubles the number of probes per sample (4 instead of 2). The role-header tokens differ by 0–2 tokens across user/assistant in production templates. Rejected — the rounded compensation values don't move; doubling probe count for a 0–2 token correction is a poor trade.

8. **Random marker samples instead of deterministic.** Would make probe results vary slightly between runs. Rejected because reproducibility across reruns of the same benchmark is more valuable than the negligible additional sample diversity.

9. **Per-request chat-template probe at request build time.** Would let the probe adapt to per-request features (tools, multimodal). Rejected because the per-request cost would be paid millions of times per benchmark; the savings of getting tools/multimodal right don't justify it (`--use-server-token-count` exists for those cases).

10. **Compensate trace-loader content by trimming `marker_tokens` from real trace text.** Would extend compensation to trace-driven benchmarks. Rejected because it changes the workload semantics — the trace is the baseline, and compensating it makes the benchmark no longer a faithful replay. If a use case ever requires this, opt-in per-loader trimming is the right interface, not a global compensation.

The current design subtracts each known-source overhead at the point in the pipeline where the corresponding wire-payload addition happens, runs all probes once at startup, never mutates user-facing config, floors defensively at 1 for pathological inputs, and matches the worker-side marker placement decision exactly. Every component has a corresponding test that asserts both the routing decision and the resulting numeric compensation.

## Test coverage map

| Concern | Test |
| --- | --- |
| Marker estimator returns 0 for `NONE` | `test_estimate_marker_token_cost_none_returns_zero` |
| Marker estimator returns positive count for active targets | `test_estimate_marker_token_cost_positive_for_active_targets` |
| Marker estimator averages across samples | `test_estimate_marker_token_cost_averages_across_samples` |
| Marker estimator rounds to int | `test_estimate_marker_token_cost_rounds_to_int` |
| Cache-bust routing: `FIRST_TURN_*` → user turn comp | `TestCacheBustMarkerRouting::test_first_turn_*` |
| Cache-bust routing: `SYSTEM_*` + shared system → no user comp | `test_system_*_with_shared_system_does_not_compensate_user_turn` |
| Cache-bust routing: `SYSTEM_*` no shared system → user turn comp (fallback) | `test_system_*_without_shared_system_compensates_first_user_turn` |
| Cache-bust routing: `NONE` → no comp anywhere | `test_none_target_compensates_nothing` |
| Shared system prompt length reduced for `SYSTEM_*` | `test_shared_system_prompt_length_reduced_for_system_prefix` |
| Shared system prompt length untouched for `FIRST_TURN_*` | `test_first_turn_target_does_not_touch_shared_system_prompt_length` |
| Shared system prompt floors at 1 for marker > length | `test_marker_larger_than_shared_system_floors_at_one` |
| User-facing config never mutated | `test_user_facing_config_is_not_mutated` |
| Probe returns `(0, 0)` for missing chat template | `test_returns_zeros_when_no_apply_chat_template` |
| Probe returns `(0, 0)` for `None` tokenizer | `test_returns_zeros_when_tokenizer_is_none` |
| Probe returns `(0, 0)` when chat template raises | `test_returns_zeros_when_apply_chat_template_raises` |
| Probe returns `(0, 0)` on negative wrap (defensive) | `test_returns_zeros_on_implausible_negative_wrap` |
| Probe correctly decomposes fixed and per-msg wrap | `test_decomposes_fixed_and_wrap` |
| `first_turn_isl_adjustment` composes all three components | `test_first_turn_adjustment_composes_all_three` |
| `subsequent_turn_isl_adjustment` only includes per-msg wrap | `test_subsequent_turn_adjustment_only_per_msg_wrap` |
| End-to-end first turn subtracts fixed + wrap + marker | `test_first_turn_subtracts_fixed_plus_wrap_plus_marker` |
| End-to-end subsequent turn subtracts only wrap | `test_subsequent_turn_subtracts_only_per_msg_wrap` |
| Floor at 1 for tiny ISL | `test_compensation_floors_at_one_for_tiny_isl` |
| Pass-through when no compensation needed | `test_no_compensation_passes_isl_through` |
| Chat template comp without cache-bust still works | `test_chat_template_only_no_cache_bust` |
| Marker estimator only invoked when needed | `test_marker_estimator_is_invoked_when_compensation_is_needed` |
