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
