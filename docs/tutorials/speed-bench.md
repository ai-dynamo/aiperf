---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with SPEED-Bench Dataset
---

# Profile with SPEED-Bench Dataset

AIPerf supports benchmarking using [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) (SPEculative Evaluation Dataset), a benchmark designed for evaluating speculative decoding across diverse semantic domains and input sequence lengths.

This guide covers profiling speculative-decoding-enabled inference servers using SPEED-Bench prompts and collecting server-side acceptance rate metrics per category.

---

## Available Dataset Variants

### Aggregate Datasets

These load all categories combined in a single dataset:

| Dataset Name | Samples | Description |
|---|---|---|
| `speed_bench_qualitative` | 880 | All 11 semantic domains combined |
| `speed_bench_throughput_1k` | 1,536 | ~1K input tokens, all 3 entropy tiers |
| `speed_bench_throughput_2k` | 1,536 | ~2K input tokens, all 3 entropy tiers |
| `speed_bench_throughput_8k` | 1,536 | ~8K input tokens, all 3 entropy tiers |
| `speed_bench_throughput_16k` | 1,536 | ~16K input tokens, all 3 entropy tiers |
| `speed_bench_throughput_32k` | 1,536 | ~32K input tokens, all 3 entropy tiers |

### Per-Category Qualitative Datasets (80 prompts each)

For per-category acceptance rate measurement, each of the 11 qualitative domains is registered separately:

| Dataset Name | Category |
|---|---|
| `speed_bench_coding` | Code generation and programming |
| `speed_bench_humanities` | History, philosophy, liberal arts |
| `speed_bench_math` | Mathematical reasoning |
| `speed_bench_multilingual` | Tasks across 23 languages |
| `speed_bench_qa` | Question answering |
| `speed_bench_rag` | Retrieval-augmented generation |
| `speed_bench_reasoning` | Logical and analytical reasoning |
| `speed_bench_roleplay` | Creative roleplay and dialogue |
| `speed_bench_stem` | Science, technology, engineering |
| `speed_bench_summarization` | Text summarization |
| `speed_bench_writing` | Creative and technical writing |

### Per-Entropy-Tier Throughput Datasets (512 prompts each)

Each throughput ISL bucket is also available filtered by entropy tier:

| Pattern | Tiers | Description |
|---|---|---|
| `speed_bench_throughput_{ISL}_low_entropy` | Code, sorting | Predictable output patterns |
| `speed_bench_throughput_{ISL}_mixed` | Needle-in-a-haystack, exams | Moderate unpredictability |
| `speed_bench_throughput_{ISL}_high_entropy` | Creative writing, dialogue | Highly unpredictable output |

Where `{ISL}` is one of: `1k`, `2k`, `8k`, `16k`, `32k`.

---

## Start a Server with Speculative Decoding

Launch an inference server with speculative decoding enabled. For example, with vLLM:

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5
```

Verify the server is ready:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Server Metrics Endpoint

AIPerf auto-discovers the Prometheus endpoint at `{url}/metrics`. If your server uses a different path, pass it explicitly with `--server-metrics`:

| Server Type | Metrics Path | Flag Needed |
|---|---|---|
| Standalone vLLM / SGLang | `/metrics` (default) | None (auto-discovered) |
| NIM-LLM containers | `/v1/metrics` | `--server-metrics http://localhost:8000/v1/metrics` |

---

## Recommended Defaults

### Non-Reasoning Models

For standard (non-reasoning) models, use `temperature=0` and a 4K output length cap:

```bash
aiperf profile \
    --model meta/llama-3.1-8b-instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset speed_bench_coding \
    --osl 4096 \
    --extra-inputs temperature:0 \
    --concurrency 16
```

Do not set `ignore_eos` — let the model stop naturally at its end-of-sequence token.

### Reasoning Models

For reasoning models (e.g., DeepSeek-R1, QwQ), follow the model card's recommended settings for temperature, top_p, and output length. Reasoning models typically require higher output limits and specific sampling parameters.

---

## Per-Category Acceptance Rate Benchmarking

To measure acceptance rates per category (matching the SPEED-Bench paper methodology), run each category separately. Each run collects speculative decoding metrics from the server's Prometheus endpoint.

### Single Category

```bash
aiperf profile \
    --model meta/llama-3.1-8b-instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset speed_bench_coding \
    --server-metrics http://localhost:8000/metrics \
    --osl 4096 \
    --extra-inputs temperature:0 \
    --concurrency 16 \
    --output-artifact-dir ./artifacts/speed_bench_coding
```

### All 11 Categories with Matrix Report

Loop through all categories, then assemble results into a per-category matrix:

```bash
CATEGORIES="coding humanities math multilingual qa rag reasoning roleplay stem summarization writing"
MODEL="meta/llama-3.1-8b-instruct"

for cat in $CATEGORIES; do
  echo "=== Running category: $cat ==="
  aiperf profile \
      --model "$MODEL" \
      --endpoint-type chat \
      --streaming \
      --url localhost:8000 \
      --public-dataset "speed_bench_${cat}" \
      --server-metrics http://localhost:8000/metrics \
      --osl 4096 \
      --extra-inputs temperature:0 \
      --concurrency 16 \
      --output-artifact-dir "./artifacts/speed_bench_${cat}"
done

# Assemble the matrix report
python scripts/speed_bench_report.py ./artifacts/ --format both
```

This produces a CSV (`speed_bench_report.csv`) and console table:

```
                         SPEED-Bench Acceptance Length Report
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Model                      ┃ coding ┃ humanities ┃ math ┃ writing ┃ Overall ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ meta/llama-3.1-8b-instruct │   1.80 │       1.84 │ 1.78 │    1.76 │    1.78 │
└────────────────────────────┴────────┴────────────┴──────┴─────────┴─────────┘
```

The report script computes acceptance length from vLLM counter metrics (`accepted_tokens / num_drafts + 1`) and also supports SGLang's direct `spec_accept_length` gauge.

Additional report metrics:

```bash
# Acceptance rate matrix (accepted / draft tokens)
python scripts/speed_bench_report.py ./artifacts/ --metric accept_rate

# Throughput matrix (output tokens/sec per category)
python scripts/speed_bench_report.py ./artifacts/ --metric throughput
```

---

## Profile with Aggregate Qualitative Split

To run all 880 prompts in a single benchmark (without per-category breakdown):

```bash
aiperf profile \
    --model meta/llama-3.1-8b-instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset speed_bench_qualitative \
    --server-metrics http://localhost:8000/metrics \
    --concurrency 16
```

---

## Profile with Throughput Splits

The throughput splits benchmark end-to-end performance at fixed input sequence lengths:

```bash
aiperf profile \
    --model meta/llama-3.1-8b-instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset speed_bench_throughput_1k \
    --server-metrics http://localhost:8000/metrics \
    --concurrency 64 \
    --benchmark-duration 120
```

Replace `speed_bench_throughput_1k` with any throughput variant (`_2k`, `_8k`, `_16k`, `_32k`) to test at different input lengths.

### Per-Entropy-Tier Throughput

To isolate entropy effects on acceptance rate at a given ISL:

```bash
for tier in low_entropy mixed high_entropy; do
  echo "=== Running throughput_1k tier: $tier ==="
  aiperf profile \
      --model meta/llama-3.1-8b-instruct \
      --endpoint-type chat \
      --streaming \
      --url localhost:8000 \
      --public-dataset "speed_bench_throughput_1k_${tier}" \
      --server-metrics http://localhost:8000/metrics \
      --concurrency 64 \
      --benchmark-duration 60
done
```

---

## Disable Server Metrics

Server metrics collection is enabled by default. To disable it:

```bash
aiperf profile \
    --model meta/llama-3.1-8b-instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset speed_bench_qualitative \
    --no-server-metrics \
    --concurrency 16
```

---

## Pre-download Dataset for Offline Use

AIPerf automatically downloads and caches the dataset on first use. To pre-download for container builds or air-gapped environments:

```bash
huggingface-cli download nvidia/SPEED-Bench --repo-type dataset
```

Or selectively download specific splits:

```python
from datasets import load_dataset

for subset in ["qualitative", "throughput_1k", "throughput_2k",
               "throughput_8k", "throughput_16k", "throughput_32k"]:
    load_dataset("nvidia/SPEED-Bench", name=subset, split="test",
                 trust_remote_code=False)
```

Set `HF_HOME` to control the cache location (e.g., `ENV HF_HOME=/opt/hf_cache` in a Dockerfile).
