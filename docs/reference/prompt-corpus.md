<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prompt corpus selection

AIPerf synthesizes prompt text from a named corpus when the dataset does not
already carry verbatim content. Author the corpus as ``prompts.corpus`` in YAML
or pass ``--prompt-corpus`` on the CLI.

## Values

| Value | Content |
|-------|---------|
| `sonnet` | Shakespeare sonnets (default for synthetic and most loaders) |
| `coding` | Procedural coding / tool-use content |
| `random` | Synthetic prompts from random vocabulary token IDs — no text file required. Matches the token-generation algorithm used by `vllm bench serve` and `sglang bench_serving`. Use with `--random-range-ratio` for ISL/OSL variance. |

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

## Prefix prompts

``coding`` uses the same synthetic prefix / shared-system / user-context
surface as ``sonnet``: those features sample from the selected corpus rather
than requiring a separate generator type.

``random`` also supports prefix prompt pools and shared system prompts. When
a prefix pool is configured, prefix prompts are generated using the same
arithmetic token-ID sequence as body prompts.

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
| SGLang | `sglang` | Full `vocab_size` range | None | Lower-bounded: `[max(1, int(mean*r)), mean]` |

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
