---
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile Embedding Models with AIPerf
---

# Profile Embedding Models with AIPerf

AIPerf supports benchmarking embedding models that convert text into dense vector representations.

This guide covers profiling OpenAI-compatible embedding endpoints using vLLM.

---

## Section 1. Profile vLLM Embedding Models

### Start a vLLM Embedding Server

Launch a vLLM server with an embedding model:

<!-- setup-vllm-embeddings-openai-endpoint-server -->
```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model BAAI/bge-small-en-v1.5
```
<!-- /setup-vllm-embeddings-openai-endpoint-server -->

Verify the server is ready:

<!-- health-check-vllm-embeddings-openai-endpoint-server -->
```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/embeddings -H "Content-Type: application/json" -d "{\"model\":\"BAAI/bge-small-en-v1.5\",\"input\":\"test\"}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```
<!-- /health-check-vllm-embeddings-openai-endpoint-server -->

### Profile with Synthetic Inputs

Run AIPerf against the embeddings endpoint using synthetic inputs:

<!-- aiperf-run-vllm-embeddings-openai-endpoint-server -->
```bash
aiperf profile \
    --model BAAI/bge-small-en-v1.5 \
    --endpoint-type embeddings \
    --endpoint /v1/embeddings \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --url localhost:8000 \
    --request-count 20 \
    --concurrency 4
```
<!-- /aiperf-run-vllm-embeddings-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is PROFILING

Profiling: 20/20 |████████████████████████| 100% [00:02<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/BAAI_bge-small-en-v1.5-embeddings-concurrency4/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                     Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│       Request Latency (ms) │  42.15 │  36.24 │  58.32 │  56.78 │  41.89 │
│  Input Sequence Length (#) │ 100.00 │ 100.00 │ 100.00 │ 100.00 │ 100.00 │
│ Request Throughput (req/s) │   9.52 │      - │      - │      - │      - │
└────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/BAAI_bge-small-en-v1.5-embeddings-concurrency4/profile_export_aiperf.json
```

Embeddings endpoints return metrics focused on request latency and throughput. No token-level metrics (TTFT, ITL) since embeddings return a single vector per request.

### Profile with Custom Input File

Create a JSONL embeddings input file and run AIPerf against it. The two
steps are combined into a single bash block so the test-docs CI actually
exercises the `aiperf profile` invocation — the runner extracts the first
bash block after the tag, so a split would leave the profile command
unrun.

<!-- aiperf-run-vllm-embeddings-openai-endpoint-server -->
```bash
cat <<EOF > inputs.jsonl
{"texts": ["What is artificial intelligence?"]}
{"texts": ["Explain machine learning."]}
{"texts": ["How do neural networks work?"]}
{"texts": ["Define deep learning."]}
{"texts": ["What are transformers in AI?"]}
EOF

aiperf profile \
    --model BAAI/bge-small-en-v1.5 \
    --endpoint-type embeddings \
    --endpoint /v1/embeddings \
    --input-file inputs.jsonl \
    --custom-dataset-type single_turn \
    --url localhost:8000 \
    --request-count 5
```
<!-- /aiperf-run-vllm-embeddings-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Loading custom dataset from inputs.jsonl
INFO     AIPerf System is PROFILING

Profiling: 5/5 |████████████████████████| 100% [00:01<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/BAAI_bge-small-en-v1.5-embeddings-custom/

JSON Export: artifacts/BAAI_bge-small-en-v1.5-embeddings-custom/profile_export_aiperf.json
```

When using custom inputs, AIPerf uses your actual text samples instead of synthetic data. The input sequence lengths will vary based on your actual text content.