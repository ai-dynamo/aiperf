{/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# Prompt corpus selection

AIPerf synthesizes prompt text from a named corpus when the dataset does not
already carry verbatim content. Author the corpus as ``prompts.corpus`` in YAML
or pass ``--prompt-corpus`` on the CLI.

## Values

| Value | Content |
|-------|---------|
| `sonnet` | Shakespeare sonnets (default for synthetic and most loaders) |
| `coding` | Procedural coding / tool-use content |
| `random` | Synthetic prompts from random vocabulary token IDs — no text file required. Matches the token-generation algorithm used by `vllm bench serve` and by `sglang.benchmark.serving` under `--dataset-name random-ids` (see note below). Use with `--random-range-ratio` for ISL/OSL variance. |

> **Which SGLang algorithm this matches.** The vocab-offset arithmetic implemented
> here is SGLang's `random_sample=False` branch, reached via
> `--dataset-name random-ids`. SGLang's *default* `--dataset-name random` is a
> different algorithm — it repeats/truncates ShareGPT token ids to hit the target
> length. Behavior is pinned against SGLang HEAD; the OSL lower bound was
> un-clamped before v0.5.x (`int(output_len * range_ratio)` with no `max(..., 1)`),
> so older SGLang releases will not reproduce these lengths.
>
> Byte-exactness also stops at the prompt round trip: AIPerf runs vLLM's
> decode → encode → trim/top-up loop, and those top-up draws consume the shared
> preseed stream. SGLang decodes once and reports `prompt_len = input_lens[i]`,
> consuming no such state — so `sglang`-style prompts diverge from SGLang
> whenever that round trip drifts.

## When it applies

Honored only where content is **synthesized**:

- synthetic datasets
- count / hash-id trace loaders (e.g. `mooncake_trace`, `bailian_trace`, `weka_trace`)
- public trace datasets that reconstruct from hash ids (e.g. SemiAnalysis weka HF)

Verbatim formats (`single_turn`, `multi_turn`, `baseten_trace`, …) ignore it.

## Defaults

When omitted, the active loader's ``default_prompt_corpus`` from the plugin
registry applies. Agentic coding loaders such as ``weka_trace`` default to
``coding``; most others default to ``sonnet``. Synthetic with no authored
corpus uses ``sonnet``.

## YAML shape

```yaml
datasets:
  - type: synthetic
    prompts:
      isl: 128
      corpus: coding

  - type: file
    format: weka_trace
    path: ./traces/
    prompts:
      corpus: coding
```

## How each corpus generates content

### sonnet

At startup `PromptGenerator` reads `assets/shakespeare.txt`, strips blank
lines, and splits the text into fixed 10,000-character chunks (deterministic
regardless of CPU count). Each chunk is tokenized in parallel and the results
are concatenated into a single flat token array.

At request time `_sample_tokens` picks a random start position in that array
via a derived RNG and returns a contiguous window of the requested length,
wrapping at the end of the corpus.

The BPE fixup loop in `generate_prompt` then validates the window:

1. Decode the token window to text.
2. Re-encode the text and compare the actual token count to the target.
3. If too long, trim tokens from the end. If too short, draw additional tokens
   from the corpus and extend.
4. Repeat up to 10 times until the re-encoded length matches the target.

A BPE-stable terminator token is probed at init time and appended to segment
boundaries so that concatenating multiple windows does not cause merge/split
drift at the join points.

### coding

`CodingContentGenerator` builds a single shuffled `_tool_pool` from roughly
22 template generators covering Python, Go, Rust, TypeScript, ML
training/inference code, bash output, JSON payloads, error tracebacks, CUDA
errors, SQL queries, git diffs, CI/CD logs, config files, markdown docs, test
output, multi-turn coding conversations, and user prompts. The approximate
weighted breakdown is: ~28% general code, ~11% ML code, ~20% bash/training
logs, ~11% JSON, ~9% errors, ~3% SQL, ~10% miscellaneous, ~8% user prompts.

All template blocks are shuffled with a seeded RNG at init time, joined with
`"\n\n"`, and tokenized once to produce the pool. At request time sampling is
identical to `sonnet` — a contiguous window with wraparound.

Unlike `sonnet`, the `coding` path does **not** run a BPE fixup loop.
`generate_prompt` decodes the sampled window and returns it directly. This
means the actual ISL may differ slightly from the target for corpora with
high BPE merge/split rates, but keeps latency low for large ISLs.

### random

No text file is loaded. For each token `j` in the request sequence:

```
token_id = allowed_tokens[(offset + request_index + j) % n]
```

where `offset` is a per-request value drawn from the RNG, `request_index`
increments across requests so successive prompts do not overlap, and `n` is
the size of the allowed token pool. The token IDs are decoded to text and
then passed through the same BPE fixup loop as `sonnet` (up to 10 retries).

See [## Random corpus](#random-corpus) below for the full explanation of
corpus style, token pool composition, and RNG alignment.

## Corpus selection resolution

`resolve_prompt_generator` in `dataset/generator/corpus.py` selects the
generator in this order:

1. Explicit `prompts.corpus` / `--prompt-corpus` value
2. Loader's `default_prompt_corpus` from the plugin registry
3. `sonnet` as the global fallback

`coding` returns a `CodingContentGenerator`; `sonnet` and `random` both
return a `PromptGenerator` with the appropriate `PromptCorpus` enum value.

## Prefix prompts

``coding`` uses the same synthetic prefix / shared-system / user-context
surface as ``sonnet``: those features sample from the selected corpus rather
than requiring a separate generator type.

``random`` also supports prefix prompt pools and shared system prompts. When
a prefix pool is configured, prefix prompts are generated using the same
arithmetic token-ID sequence as body prompts.

### Token-level prefix concatenation

The prefix is joined to the body at the **token-ID level**, and the combined
sequence goes through a single decode/re-encode correction targeting exactly
`prefix_len + body_len` — the same contract as vLLM's `generate_token_sequence`.
A prompt therefore contains exactly the configured number of tokens.

Joining the two decoded segments as strings instead (`f"{prefix} {body}"`) would
let BPE charge for the separator: measured on GPT-2 that cost one extra token on
~81% of requests, +0.84 on average, and it perturbed the per-request token index.
See AIP-1118.

For the RANDOM corpus the pool is drawn from the shared preseed stream *after*
the ISL/OSL/offset draws, bounded by `len(allowed_tokens)`, mirroring vLLM's
`RandomDataset.get_prefix`. Byte-exact prompt parity with vLLM requires
`--prompt-prefix-pool-size 1`, since vLLM has exactly one prefix; larger pools
consume additional draws from the shared stream and so diverge in the
(rarely-exercised) top-up token values.

### Prefix semantics vs vLLM / SGLang

The prefix is **additive** in every corpus, matching the reference benchmarkers:

| Tool | Prefix behaviour | Wire ISL |
|------|-----------------|----------|
| aiperf (`--prompt-prefix-pool-size` / `--prompt-prefix-length`) | Prefix is **additive** — body tokens = configured ISL, prefix prepended on top | configured ISL + prefix_len |
| vLLM (`--random-prefix-len`) | Same — additive | configured ISL + prefix_len |
| SGLang (`prefix_len`) | Same — additive | configured ISL + prefix_len |

So `--prompt-input-tokens-mean 128 --prompt-prefix-length 20` produces 148-token
prompts, not 128. `--prompt-input-tokens-mean` describes the *body*; the prefix
represents cached shared context riding on top of it, which is what makes
prefix pools useful for KV-cache benchmarking.

Note this differs from the special-token and chat-template compensation, which
*are* subtracted from the ISL target so the wire length lands on the configured
value. Only the prefix is additive.

The two compensations are applied by different mechanisms, and every corpus
style applies both. Special tokens come off the window bounds under `vllm`
style and off each drawn length under `sglang` style; chat-template wrapping
always comes off per-request in the composer. See
[ISL budget compensation](./isl-budget-compensation.md).

To keep a fixed total instead, subtract the prefix length yourself:
`--prompt-input-tokens-mean 108 --prompt-prefix-length 20` gives 128 on the wire.

## Random corpus

``random`` generates prompts entirely from vocabulary token IDs using the
formula `(offset + request_index + j) % len(allowed_tokens)` for each token
`j` in the sequence, then decodes to text and re-encodes to verify the
round-trip token count (BPE fixup, up to 10 retries). No text file is loaded.

### Corpus style

``random`` is paired with `--random-corpus-style` (or `random_corpus_style`
in YAML) to select which benchmarking tool's behavior to replicate:

| Style | `--random-corpus-style` | Token pool | BOS adjustment | Range formula |
|-------|------------------------|-----------|---------------|---------------|
| vLLM (default) | `vllm` | Non-special tokens only | Subtract BOS from ISL mean | Symmetric: `[floor(mean*(1-r)), ceil(mean*(1+r))]` |
| SGLang | `sglang` | Full `vocab_size` range | Subtract per-request after sampling | Lower-bounded: `[max(1, int(mean*r)), mean]` |

### RNG alignment (vLLM style)

When `--random-corpus-style vllm` and `--random-seed` are set, aiperf
pre-generates all ISL values, then all OSL values, then all per-request
offsets from a single `numpy.random.default_rng(seed)` — matching vLLM's
`get_sampling_params` draw order. This ensures identical token sequences for
the same seed when comparing aiperf against `vllm bench serve`.

### Example

```yaml
datasets:
  - type: synthetic
    prompts:
      isl: 128
      osl: 128
      corpus: random
      random_range_ratio: "0.3"
      random_corpus_style: vllm
```

```bash
aiperf profile \
  --prompt-corpus random \
  --random-range-ratio 0.3 \
  --random-corpus-style vllm \
  --prompt-input-tokens-mean 128 \
  --prompt-output-tokens-mean 128 \
  --random-seed 0
```
