<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile Hugging Face TGI Models with AIPerf

AIPerf can benchmark **Large Language Models (LLMs)** served through the
[Hugging Face Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference)
`generate` API.
TGI exposes two standard HTTP endpoints for text generation:

| Endpoint | Description | AIPerf Flag |
|-----------|--------------|--------------|
| `/generate` | Returns the full text completion in one response (non-streaming). | *(default)* |
| `/generate_stream` | Streams generated tokens as they are produced (SSE). | `--streaming` |


## Start a Hugging Face TGI Server

To launch a Hugging Face TGI server, use the official `ghcr.io` image:

```bash
docker run --gpus all --rm -it \
  -p 8080:80 \
  -e MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  ghcr.io/huggingface/text-generation-inference:latest
```

```bash
# Verify the server is running
curl -s http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs":"Hello world"}' | jq
```

## Profile with AIPerf

You can benchmark TGI models in either non-streaming or streaming,
and with either synthetic inputs or a custom input file.

### Non-Streaming (`/generate`)

#### Profile with synthetic inputs

```bash
aiperf profile \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --endpoint-type huggingface_generate \
    --url localhost:8080 \
    --request-count 10
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Using Hugging Face TGI /generate endpoint (non-streaming)
INFO     AIPerf System is PROFILING

Profiling: 10/10 |████████████████████████| 100% [00:08<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/TinyLlama_TinyLlama-1.1B-Chat-v1.0-generate-concurrency1/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1234.56 │ 987.34 │ 1567.89 │ 1567.89 │ 1198.45 │
│ Output Token Count (tokens) │  256.00 │ 200.00 │  300.00 │  300.00 │  254.00 │
│  Request Throughput (req/s) │    2.34 │      - │       - │       - │       - │
└─────────────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/TinyLlama_TinyLlama-1.1B-Chat-v1.0-generate-concurrency1/profile_export_aiperf.json
```

#### Profile with custom input file

You can also provide your own text prompts using the
--input-file option.
The file should be in JSONL format and contain text entries.

```bash
cat > inputs.jsonl <<'EOF'
{"text": "Hello TinyLlama!"}
{"text": "Tell me a joke."}
EOF
```
Then run:

```bash
aiperf profile \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --endpoint-type huggingface_generate \
    --url localhost:8080 \
    --input-file ./inputs.jsonl \
    --custom-dataset-type single_turn \
    --request-count 10
```

### Streaming (`/generate_stream`)

When the `--streaming` flag is enabled, AIPerf automatically sends requests to the `/generate_stream` endpoint of the TGI server.

#### Profile with synthetic inputs

```bash
aiperf profile \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --endpoint-type huggingface_generate \
    --url localhost:8080 \
    --streaming \
    --request-count 10
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Using Hugging Face TGI /generate_stream endpoint (streaming)
INFO     AIPerf System is PROFILING

Profiling: 10/10 |████████████████████████| 100% [00:09<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/TinyLlama_TinyLlama-1.1B-Chat-v1.0-generate-concurrency1/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃    min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 1189.45 │ 945.67 │ 1498.34 │ 1498.34 │ 1156.78 │
│    Time to First Token (ms) │  234.56 │ 189.34 │  298.45 │  298.45 │  228.90 │
│    Inter Token Latency (ms) │   14.23 │  11.45 │   18.90 │   18.90 │   13.89 │
│ Output Token Count (tokens) │  256.00 │ 200.00 │  300.00 │  300.00 │  254.00 │
│  Request Throughput (req/s) │    2.56 │      - │       - │       - │       - │
└─────────────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/TinyLlama_TinyLlama-1.1B-Chat-v1.0-generate-concurrency1/profile_export_aiperf.json
```

#### Profile with custom input file

Create your own prompt file in JSONL format:

```bash
cat > inputs.jsonl <<'EOF'
{"text": "Explain quantum computing in simple terms."}
{"text": "Write a haiku about rain."}
{"text": "Summarize the causes of the French Revolution."}
EOF
```

Then run:

```bash
aiperf profile \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --endpoint-type huggingface_generate \
    --url localhost:8080 \
    --input-file ./inputs.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --request-count 10
```
