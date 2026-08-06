---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Inline Datasets
---

# Inline Datasets

Embed your benchmark dataset directly in the YAML config — no separate JSONL file required.

## When to inline vs. when to keep a file

| | Inline (`records:`) | File (`path:`) |
|---|---|---|
| Few records (< ~100) | Recommended | OK |
| Many records (> ~500) | Discouraged (warning emitted) | Recommended |
| Single-file deployment unit (k8s ConfigMap) | Recommended | Requires sidecar mount |
| Shareable repro for a colleague | Recommended | Two files to ship |
| Records updated independently of the config | Discouraged | Recommended |

The schema is the same: each inline record matches one line of the equivalent JSONL file.

## Single-turn

```yaml
schemaVersion: "2.0"
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000/v1/chat/completions
  dataset:
    type: file
    format: single_turn
    records:
      - {text: "What is machine learning?"}
      - {text: "Explain GANs in two sentences.", output_length: 200}
      - {text: "Define reinforcement learning."}
  phases:
    type: concurrency
    concurrency: 2
    requests: 100
```

## Multi-turn

```yaml
benchmark:
  dataset:
    type: file
    format: multi_turn
    records:
      - session_id: chat_1
        turns:
          - {text: "What is machine learning?"}
          - {text: "Can you give me an example?"}
      - session_id: chat_2
        turns:
          - {text: "Explain neural networks."}
          - {text: "How do they differ from traditional algorithms?"}
          - {text: "Which architecture for image classification?"}
```

## Random pool

Single-pool inline:

```yaml
benchmark:
  dataset:
    type: file
    format: random_pool
    sampling: random
    records:
      - {text: "Common query", type: random_pool}
      - {text: "Less common query", type: random_pool}
      - {text: "Rare query", type: random_pool}
```

Multi-pool inline (mirrors a directory-of-JSONLs file layout):

```yaml
benchmark:
  dataset:
    type: file
    format: random_pool
    records:
      queries:
        - {text: "What is your refund policy?", type: random_pool}
        - {text: "How do I reset my password?", type: random_pool}
      passages:
        - {text: "Refunds are processed within 5 business days.", type: random_pool}
        - {text: "Click 'Forgot password' on the login page.", type: random_pool}
```

## Trace replay (mooncake_trace)

```yaml
benchmark:
  dataset:
    type: file
    format: mooncake_trace
    synthesis:
      speedup_ratio: 2.0    # replay 2x faster (1.0 = real-time, 0.5 = 2x slower)
    records:
      - {timestamp: 0,    input_length: 512,  output_length: 128, hash_ids: [1, 2, 3]}
      - {timestamp: 100,  input_length: 1024, output_length: 256, hash_ids: [4, 5]}
      - {timestamp: 250,  input_length: 256,  output_length: 64,  hash_ids: [1, 2]}
```

## Mutual exclusion

`path:` and `records:` are mutually exclusive. Setting both, or neither, raises a Pydantic `ValidationError` at config load with this message:

```text
FileDataset requires exactly one source: set either `path:` (load from disk) or `records:` (embed in YAML), not both. Got path=<...>, records=<...>.
```

## Soft size limit

If you inline more than 500 records, AIPerf logs a warning recommending a file. There is no hard cap — you can keep going if you have a good reason — but reading a 5,000-line YAML in code review is rough. The threshold is configurable via `AIPERF_DATASET_INLINE_RECORDS_WARN_THRESHOLD`.

## Tutorial template

A bundled template demonstrates all three formats:

```bash
aiperf config init --template inline_dataset --output bench.yaml
```
