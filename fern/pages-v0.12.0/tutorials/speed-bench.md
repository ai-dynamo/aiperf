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

## Prepare the Dataset

NOTICE: This dataset is governed by the [NVIDIA Evaluation Dataset License Agreement](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/main/License.pdf). For each dataset a user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose. The prepare data script below automatically fetches data from all the source datasets.

You should first download and prepare the dataset using the following one liner:

```bash
SPEED_BENCH_DIR="./datasets/speed-bench"
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py | python3 - --output_dir $SPEED_BENCH_DIR
```

This will download all splits into the working directory as JSONL files. Other supported options of the prepare script:

* `--config`: select which config to prepare, can be one of the splits in the dataset (e.g., `qualitative`, `throughput_2k`) or `all` to prepare all of the configs.
* `--output_dir`: select different output directory to download the dataset to.

---

## Start a Server with Speculative Decoding

Launch an inference server with speculative decoding enabled. For example, with vLLM:

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5, "method": "draft_model"}'
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
    --model meta-llama/Llama-3.1-8B-Instruct\
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --custom-dataset-type speed_bench_coding \
    --input-file ${SPEED_BENCH_DIR}/qualitative.jsonl \
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
    --custom-dataset-type speed_bench_coding \
    --input-file ${SPEED_BENCH_DIR}/qualitative.jsonl \
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
      --custom-dataset-type speed_bench_${cat} \
      --input-file ${SPEED_BENCH_DIR}/qualitative.jsonl \
      --server-metrics http://localhost:8000/metrics \
      --osl 4096 \
      --extra-inputs temperature:0 \
      --concurrency 16 \
      --output-artifact-dir "./artifacts/speed_bench_${cat}"
done

# Assemble the matrix report
aiperf speed-bench-report ./artifacts/ --format both
```

This produces a CSV (`speed_bench_report.csv`) and console table:

```
                         Acceptance Length Report
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
aiperf speed-bench-report ./artifacts/ --metric accept_rate

# Throughput matrix (output tokens/sec per category)
aiperf speed-bench-report ./artifacts/ --metric throughput
```

---

## Literature Acceptance-Length Datasets (GSM8K, MT-Bench, MATH-500, HumanEval, MBPP)

The speculative-decoding literature overwhelmingly reports acceptance length against five standard benchmarks. AIPerf registers each as a public dataset that is auto-downloaded from HuggingFace at runtime, so there is no prepare-data step: just select one with `--public-dataset` and run the same `aiperf speed-bench-report` workflow shown above.

| Dataset Name | HuggingFace Source | Prompts | Turns | License |
|---|---|---|---|---|
| `spec_al_gsm8k` | `openai/gsm8k` (`main`, `test`) | 1,319 | single | MIT |
| `spec_al_math500` | `HuggingFaceH4/MATH-500` (`test`) | 500 | single | MIT |
| `spec_al_humaneval` | `openai/openai_humaneval` (`test`) | 164 | single | MIT |
| `spec_al_mbpp` | `google-research-datasets/mbpp` (`full`, `test`) | 500 | single | CC-BY-4.0 |
| `spec_al_mtbench` | `HuggingFaceH4/mt_bench_prompts` (`train`) | 80 | two-turn | Apache-2.0 |

Prompts are emitted verbatim (the raw question/problem/prompt field); the served model's chat template wraps them at request time via `--endpoint-type chat`. HumanEval and MBPP are text-completion tasks in the spec-decode literature, so chat-wrapping them keeps the matrix uniform but shifts their acceptance length somewhat from the papers' headline numbers. Acceptance length is correctness-agnostic, so use greedy decoding (`--extra-inputs temperature:0`) to match the headline numbers reported in the literature. Note that `--osl` does not apply to public datasets, so cap generation with `--extra-inputs max_tokens:N` instead. `spec_al_mtbench` is multi-turn: AIPerf dispatches both turns per session and feeds the live assistant reply back as conversation history between them - size it with `--num-conversations` rather than `--request-count` (see below).

### Run All Five with a Matrix Report

```bash
MODEL="meta/llama-3.1-8b-instruct"
ART=./artifacts/spec-al   # dedicated root so this matrix never merges with speed_bench_* runs

# Single-turn datasets: size each run to the full dataset with --request-count.
for pair in spec_al_gsm8k:1319 spec_al_math500:500 spec_al_humaneval:164 spec_al_mbpp:500; do
  ds="${pair%%:*}"; count="${pair##*:}"
  echo "=== Running dataset: $ds ($count requests) ==="
  aiperf profile \
      --model "$MODEL" \
      --endpoint-type chat \
      --streaming \
      --url localhost:8000 \
      --public-dataset "$ds" \
      --server-metrics http://localhost:8000/metrics \
      --request-count "$count" \
      --extra-inputs temperature:0 max_tokens:4096 \
      --concurrency 16 \
      --output-artifact-dir "$ART/$ds"
done

# MT-Bench is multi-turn (80 two-turn conversations). Size it with
# --num-conversations so every session runs exactly once; --request-count
# recycles the 80 sessions to reach the count and would dispatch each prompt
# more than once.
aiperf profile \
    --model "$MODEL" \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset spec_al_mtbench \
    --server-metrics http://localhost:8000/metrics \
    --num-conversations 80 \
    --extra-inputs temperature:0 max_tokens:4096 \
    --concurrency 16 \
    --output-artifact-dir "$ART/spec_al_mtbench"

# Assemble the acceptance-length matrix (one column per dataset)
aiperf speed-bench-report "$ART" --metric accept_length --format both
```

> Size each run to the full dataset — without an explicit count AIPerf defaults
> to 10 requests. Single-turn datasets use `--request-count`; the multi-turn
> `spec_al_mtbench` uses `--num-conversations 80` (one run per conversation),
> since `--request-count` recycles its 80 sessions to reach the count. Cap
> generation with `--extra-inputs max_tokens:N` (`--osl` is ignored for public
> datasets), and keep these runs in their own artifacts directory so
> `speed-bench-report` does not average them into an unrelated `speed_bench_*`
> matrix.

The report recognizes these runs the same way it recognizes the `speed_bench_*` runs, producing one matrix column per dataset:

```
                         Acceptance Length Report
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ Model                      ┃ gsm8k ┃ math500 ┃ mtbench ┃ humaneval ┃ mbpp ┃ Overall ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━┩
│ meta/llama-3.1-8b-instruct │  2.40 │    2.31 │    1.95 │      2.62 │ 2.55 │    2.37 │
└────────────────────────────┴───────┴─────────┴─────────┴───────────┴──────┴─────────┘
```

The `accept_rate` and `throughput` metrics work identically (`aiperf speed-bench-report ./artifacts/ --metric accept_rate`).

---

## Profile with Aggregate Qualitative Split

To run all 880 prompts in a single benchmark (without per-category breakdown):

```bash
aiperf profile \
    --model meta/llama-3.1-8b-instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --custom-dataset-type speed_bench_qualitative \
    --input-file ${SPEED_BENCH_DIR}/qualitative.jsonl \
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
    --custom-dataset-type speed_bench_throughput_1k \
    --input-file ${SPEED_BENCH_DIR}/throughput_1k.jsonl \
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
      --custom-dataset-type "speed_bench_throughput_1k_${tier}" \
      --input-file ${SPEED_BENCH_DIR}/throughput_1k.jsonl \
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
    --custom-dataset-type speed_bench_qualitative \
    --input-file ${SPEED_BENCH_DIR}/qualitative.jsonl \
    --no-server-metrics \
    --concurrency 16
```
