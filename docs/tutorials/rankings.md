<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile Ranking Models with AIPerf

AIPerf supports benchmarking **ranking and reranking models**, including those served through
**Hugging Face Text Embeddings Inference (TEI)** or **Cohere Re-Rank APIs**.
These models take a query and one or more passages, returning a similarity or relevance score.

---

## Section 1. Profile Hugging Face TEI Re-Rank Models

### Start a Hugging Face TEI Server

Launch a Hugging Face Text Embeddings Inference (TEI) container in re-ranker mode:

```bash
docker run --gpus all --rm -it \
  -p 8080:80 \
  -e MODEL_ID=BAAI/bge-reranker-base \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-reranker-base --port 80
```

```bash
# Verify server is running
curl -s http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"What is AI?", "texts":["AI is artificial intelligence.","Bananas are yellow."]}' | jq
```

### Profile using Synthetic Inputs

Run AIPerf using the following command:
```bash
aiperf profile \
    -m BAAI/bge-reranker-base \
    --endpoint-type hf_tei_rankings \
    --url localhost:8080 \
    --request-count 10 \
    --rankings-passages-mean 5 \
    --rankings-passages-stddev 1 \
    --rankings-passages-prompt-token-mean 32 \
    --rankings-passages-prompt-token-stddev 8 \
    --rankings-query-prompt-token-mean 16 \
    --rankings-query-prompt-token-stddev 4
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is PROFILING

Profiling: 10/10 |████████████████████████| 100% [00:02<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/BAAI_bge-reranker-base-rankings/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃                     Metric ┃   avg ┃   min ┃   max ┃   p99 ┃   p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│       Request Latency (ms) │ 52.34 │ 45.12 │ 68.45 │ 65.23 │ 51.89 │
│ Request Throughput (req/s) │  5.12 │     - │     - │     - │     - │
└────────────────────────────┴───────┴───────┴───────┴───────┴───────┘

JSON Export: artifacts/BAAI_bge-reranker-base-rankings/profile_export_aiperf.json
```

> **Note:** The rankings-specific token options cannot be used together with `--prompt-input-tokens-mean` or `--prompt-input-tokens-stddev`. Use the rankings-specific options for controlling token counts in rankings queries and passages.

### Profile using Custom Inputs

Create a file named rankings.jsonl where each line represents a ranking request with a query and one or more passages.

```bash
cat <<EOF > rankings.jsonl
{"texts":[{"name":"query","contents":["What is AI topic 0?"]},{"name":"passages","contents":["AI passage 0"]}]}
{"texts":[{"name":"query","contents":["What is AI topic 1?"]},{"name":"passages","contents":["AI passage 1"]}]}
{"texts":[{"name":"query","contents":["What is AI topic 2?"]},{"name":"passages","contents":["AI passage 2"]}]}
{"texts":[{"name":"query","contents":["What is AI topic 3?"]},{"name":"passages","contents":["AI passage 3"]}]}
{"texts":[{"name":"query","contents":["What is AI topic 4?"]},{"name":"passages","contents":["AI passage 4"]}]}
EOF
```

Run AIPerf using the following command:
```bash
aiperf profile \
    -m BAAI/bge-reranker-base \
    --endpoint-type hf_tei_rankings \
    --url localhost:8080 \
    --input-file ./rankings.jsonl \
    --custom-dataset-type single_turn \
    --request-count 10
```

## Section 2. Profile Cohere Re-Rank API

### Start vLLM Server in Cohere Mode

Run vLLM with the `--runner` pooling flag to enable reranking behavior:

```bash
docker run --gpus all -p 8080:8000 \
  -e HF_TOKEN=<HF_TOKEN> \
  vllm/vllm-openai:latest \
  --model BAAI/bge-reranker-v2-m3 \
  --runner pooling
```

```bash
# Verify the server
curl -s http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"What is AI?","documents":["Artificial intelligence overview","Bananas are yellow"]}' | jq
```

### Profile using Synthetic Inputs
Run AIPerf using the following command:

```bash
aiperf profile \
    -m BAAI/bge-reranker-v2-m3 \
    --endpoint-type cohere_rankings \
    --url localhost:8080 \
    --request-count 10
```

### Profile using Custom Inputs

Create a file named `rankings.jsonl`:
```bash
cat <<EOF > rankings.jsonl
{"texts":[{"name":"query","contents":["What is AI topic 0?"]},{"name":"passages","contents":["AI passage 0"]}]}
{"texts":[{"name":"query","contents":["What is AI topic 1?"]},{"name":"passages","contents":["AI passage 1"]}]}
{"texts":[{"name":"query","contents":["What is AI topic 2?"]},{"name":"passages","contents":["AI passage 2"]}]}
{"texts":[{"name":"query","contents":["What is AI topic 3?"]},{"name":"passages","contents":["AI passage 3"]}]}
{"texts":[{"name":"query","contents":["What is AI topic 4?"]},{"name":"passages","contents":["AI passage 4"]}]}
EOF
```

Run AIPerf:

```bash
aiperf profile \
    -m BAAI/bge-reranker-v2-m3 \
    --endpoint-type cohere_rankings \
    --url localhost:8080 \
    --input-file ./rankings.jsonl \
    --custom-dataset-type single_turn \
    --request-count 10
```
