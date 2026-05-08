---
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile Ranking Models with AIPerf
---

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

> [!NOTE]
> The rankings-specific token options cannot be used together with `--prompt-input-tokens-mean` or `--prompt-input-tokens-stddev`. Use the rankings-specific options for controlling token counts in rankings queries and passages.

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

### Profile vLLM Vision Rerank Models

The `cohere_rankings` endpoint also supports vLLM multimodal rerank documents.
Text-only requests keep the standard Cohere shape, while multimodal requests use
structured documents with `content` parts for text and media. AIPerf pairs
`passages`, `images`, and `videos` by index, so each non-empty modality must have
the same number of entries.

For synthetic rankings inputs, AIPerf generates one image or video per synthetic
passage when image or video generation is enabled. This keeps the generated
multimodal documents index-paired.

Run AIPerf with synthetic multimodal rankings inputs:

```bash
aiperf profile \
    -m nvidia/llama-nemotron-rerank-vl-1b-v2 \
    --endpoint-type cohere_rankings \
    --custom-endpoint /rerank \
    --url localhost:8080 \
    --request-count 10 \
    --rankings-passages-mean 4 \
    --rankings-passages-stddev 0 \
    --rankings-passages-prompt-token-mean 32 \
    --rankings-passages-prompt-token-stddev 0 \
    --rankings-query-prompt-token-mean 16 \
    --rankings-query-prompt-token-stddev 0 \
    --image-width-mean 224 \
    --image-width-stddev 0 \
    --image-height-mean 224 \
    --image-height-stddev 0 \
    --image-batch-size 1
```

You can supply images as `data:` URLs (as in the example below), using the usual
`data:image/<subtype>;base64,<payload>` form, or as absolute `http://` or
`https://` URLs to a reachable image resource when the rerank server fetches media
from the network.

To use custom multimodal inputs, create a rankings file:

```bash
cat <<EOF > multimodal-rankings.jsonl
{"texts":[{"name":"query","contents":["Retrieve the beach image"]},{"name":"passages","contents":["A beach at sunset"]}],"images":[{"name":"image_url","contents":["data:image/png;base64,<BASE64_IMAGE>"]}]}
{"texts":[{"name":"query","contents":["Retrieve the skyline image"]},{"name":"passages","contents":["A city skyline at night"]}],"images":[{"name":"image_url","contents":["data:image/png;base64,<BASE64_IMAGE>"]}]}
EOF
```

Run AIPerf against a vLLM server.

```bash
aiperf profile \
    -m nvidia/llama-nemotron-rerank-vl-1b-v2 \
    --endpoint-type cohere_rankings \
    --url localhost:8080 \
    --input-file ./multimodal-rankings.jsonl \
    --custom-dataset-type single_turn \
    --request-count 10
```
