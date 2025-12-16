<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Mock Server

A mock server for integration testing and performance benchmarking of LLM applications. Provides OpenAI-compatible APIs, multi-backend Prometheus metrics simulation, configurable latency, and deterministic responses.

## Features

- [**OpenAI API Compatibility**](#openai-compatible-endpoints): Chat completions, text completions, and embeddings
- [**Multi-Backend Ranking**](#ranking-endpoints): NVIDIA NIM, HuggingFace TEI, and Cohere reranking
- [**HuggingFace TGI Support**](#huggingface-tgi-endpoints): `/generate` and `/generate_stream` endpoints
- [**Prometheus Metrics**](#prometheus-metrics): vLLM, SGLang, TensorRT-LLM, and NVIDIA Dynamo backends
- [**GPU Telemetry**](#gpu-telemetry): Simulated DCGM metrics for multiple GPU types
- [**Configurable Latency**](#latency-options): TTFT, ITL, and per-endpoint latency models
- [**Reasoning Models**](#reasoning-models): GPT-OSS/Qwen-style reasoning with `reasoning_content`
- [**Error Injection**](#error-injection): Reproducible fault testing with configurable error rates
- [**Deterministic Responses**](#token-generation): Hash-based generation for identical outputs
- [**Fast Mode**](#quick-start): Zero-latency mode (`--fast`) for integration testing
- [**Corpus**](#corpus): Pre-tokenized corpus loaded from aiperf's shakespeare.txt for deterministic output

### Supported Endpoints

**Inference APIs**

| Endpoint | Description |
|----------|-------------|
| [`/v1/chat/completions`](#chat-completions) | OpenAI chat completions (streaming supported) |
| [`/v1/completions`](#text-completions) | OpenAI text completions (streaming supported) |
| [`/v1/embeddings`](#embeddings) | OpenAI embeddings (768-dim) |
| [`/v1/ranking`](#nvidia-nim-ranking) | NVIDIA NIM ranking |
| [`/rerank`](#huggingface-tei-rerank) | HuggingFace TEI reranking |
| [`/v2/rerank`](#cohere-rerank) | Cohere reranking |
| [`/generate`](#generate-non-streaming) | HuggingFace TGI generation |
| [`/generate_stream`](#generate-stream) | HuggingFace TGI streaming |
| [`/v1/custom-multimodal`](#custom-multimodal-endpoint) | Custom multimodal format |

**Prometheus Metrics**

| Endpoint | Description |
|----------|-------------|
| [`/metrics`](#prometheus-metrics) | Mock server metrics |
| [`/vllm/metrics`](#prometheus-metrics) | vLLM-compatible |
| [`/sglang/metrics`](#prometheus-metrics) | SGLang-compatible |
| [`/trtllm/metrics`](#prometheus-metrics) | TensorRT-LLM-compatible |
| [`/dynamo_frontend/metrics`](#prometheus-metrics) | Dynamo frontend |
| [`/dynamo_component/prefill/metrics`](#prometheus-metrics) | Dynamo prefill worker |
| [`/dynamo_component/decode/metrics`](#prometheus-metrics) | Dynamo decode worker |

**GPU Telemetry**

| Endpoint | Description |
|----------|-------------|
| [`/dcgm1/metrics`](#gpu-telemetry) | DCGM GPU metrics (instance 1) |
| [`/dcgm2/metrics`](#gpu-telemetry) | DCGM GPU metrics (instance 2) |

**Health & Info**

| Endpoint | Description |
|----------|-------------|
| [`/health`](#health--info) | Health check with config |
| [`/`](#health--info) | Server info and version |

## Installation

```bash
# From project root
make install-mock-server

# Or standalone
cd tests/aiperf_mock_server
pip install -e ".[dev]"
```

## Quick Start

```bash
# Start with defaults
aiperf-mock-server

# Or run as module
python -m aiperf_mock_server

# Fast mode for integration testing (zero latency)
aiperf-mock-server --fast

# Custom configuration
aiperf-mock-server --port 8080 --ttft 50 --itl 10

# Verbose logging
aiperf-mock-server -v
```

## Configuration

Configuration via CLI arguments or environment variables (`MOCK_SERVER_` prefix).

### Server Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--port` | `-p` | `8000` | Server port |
| `--host` | | `127.0.0.1` | Bind address |
| `--workers` | `-w` | `1` | Uvicorn worker count |
| `--fast` | `-f` | `false` | Zero latency mode |
| `--log-level` | | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `--verbose` | `-v` | `false` | Debug logging (overrides log-level) |
| `--access-logs` | | `false` | HTTP access logs |

### Latency Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--ttft` | `-t` | `20.0` | Time to first token (ms) |
| `--itl` | | `5.0` | Inter-token latency (ms) |
| `--embedding-base-latency` | | `10.0` | Embedding base latency (ms) |
| `--embedding-per-input-latency` | | `2.0` | Per-input latency (ms) |
| `--ranking-base-latency` | | `10.0` | Ranking base latency (ms) |
| `--ranking-per-passage-latency` | | `1.0` | Per-passage latency (ms) |

### Error Injection

| Option | Default | Description |
|--------|---------|-------------|
| `--error-rate` | `0.0` | Error injection rate (0-100%) |
| `--random-seed` | `None` | Seed for reproducible errors |

### GPU Telemetry

| Option | Default | Description |
|--------|---------|-------------|
| `--dcgm-gpu-name` | `h200` | GPU model (`rtx6000`, `a100`, `h100`, `h100-sxm`, `h200`, `b200`, `gb200`) |
| `--dcgm-num-gpus` | `2` | Number of GPUs (1-8) |
| `--dcgm-auto-load` | `true` | Auto-scale DCGM load based on token throughput |
| `--dcgm-min-throughput` | `100` | Minimum baseline tokens/sec (auto-scales above) |
| `--dcgm-window-sec` | `1.0` | Throughput sliding window (0.1-60 seconds) |
| `--dcgm-hostname` | `localhost` | Hostname in metrics |
| `--dcgm-seed` | `None` | Seed for deterministic metrics |

### Tokenizer Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tokenizer` | `Qwen/Qwen3-0.6B` | HuggingFace tokenizer for corpus |
| `--tokenizer-revision` | `main` | Tokenizer revision (branch, tag, or commit) |
| `--tokenizer-trust-remote-code` | `false` | Trust remote code for custom tokenizers |
| `--no-tokenizer` | `false` | Skip tokenizer, use character-based chunking (faster startup) |

**Auto-Scaling GPU Metrics**

DCGM metrics automatically scale based on observed token throughput. The system tracks peak throughput and uses that as 100% load:

```
Token Flow:
┌─────────────────┐
│  LLM Endpoint   │  (chat/completions)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Token Generation   │  → record_streamed_token() / record_token_metrics()
└────────┬────────────┘
         │                ┌─────────────────────────────────────┐
         │                │ _token_events (1-sec sliding window)│
         │                │ throughput = tokens / window        │
         │                │ max_observed = max(max, throughput) │
         │                │ load = throughput / max_observed    │
         │                └────────┬────────────────────────────┘
         │                         │
         │                         ▼
         │                ┌──────────────────┐
         │                │  DCGMFaker(s)    │  → set_load(load)
         │                │  .generate()     │  → GPU metrics reflect load
         │                └──────────────────┘
         │
         ▼
┌─────────────────┐
│    Response     │
└─────────────────┘
```

**Behavior:**
- Tracks peak observed throughput automatically
- Current throughput / peak = GPU load percentage
- `--dcgm-min-throughput` sets a floor (prevents div-by-tiny-number during warmup)
- No manual tuning needed - adapts to your workload

### Environment Variables

```bash
export MOCK_SERVER_PORT=8080
export MOCK_SERVER_FAST=true
aiperf-mock-server
```

## API Endpoints

### OpenAI-Compatible Endpoints

#### Chat Completions

**`POST /v1/chat/completions`**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

**Parameters:**
- `model` (required): Model identifier
- `messages` (required): Array of `{role, content}` objects
- `max_completion_tokens` or `max_tokens`: Maximum output tokens
- `stream`: Enable streaming (default: false)
- `stream_options`: `{"include_usage": true}` for usage in stream
- `reasoning_effort`: `"low"` | `"medium"` | `"high"` (for reasoning models)
- `min_tokens`: Minimum tokens to generate
- `ignore_eos`: Generate exactly `max_tokens`

#### Text Completions

**`POST /v1/completions`**

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "prompt": "Hello world", "max_tokens": 50}'
```

**Parameters:**
- `model` (required): Model identifier
- `prompt` (required): String or array of strings
- `max_tokens`, `stream`, `stream_options`, `min_tokens`, `ignore_eos`: Same as chat

#### Embeddings

**`POST /v1/embeddings`**

Returns deterministic 768-dimensional embeddings based on input hash.

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding", "input": ["text1", "text2"]}'
```

**Parameters:**
- `model` (required): Model identifier
- `input` (required): String or array of strings

### Ranking Endpoints

Three ranking endpoint formats are supported:

#### NVIDIA NIM Ranking

**`POST /v1/ranking`**

```bash
curl -X POST http://localhost:8000/v1/ranking \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reranker",
    "query": {"text": "query"},
    "passages": [{"text": "passage1"}, {"text": "passage2"}]
  }'
```

#### HuggingFace TEI Rerank

**`POST /rerank`**

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "query", "texts": ["text1", "text2"]}'
```

#### Cohere Rerank

**`POST /v2/rerank`**

```bash
curl -X POST http://localhost:8000/v2/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "query", "documents": ["doc1", "doc2"]}'
```

### HuggingFace TGI Endpoints

#### Generate (Non-Streaming)

**`POST /generate`**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello world", "parameters": {"max_new_tokens": 50}}'
```

#### Generate Stream

**`POST /generate_stream`**

```bash
curl -N -X POST http://localhost:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello world", "parameters": {"max_new_tokens": 50}}'
```

### Custom Multimodal Endpoint

**`POST /v1/custom-multimodal`**

Example custom format for testing non-standard APIs:

```bash
curl -X POST http://localhost:8000/v1/custom-multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "modality_bundle": {
      "text_fragments": ["text1"],
      "visual_assets": {"images": [], "videos": []},
      "audio_streams": []
    },
    "inference_params": {"model_id": "multimodal-model"}
  }'
```

### Health & Info

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check with config |
| `GET /` | Server info and version |

### GPU Telemetry

**`GET /dcgm1/metrics`** and **`GET /dcgm2/metrics`**

DCGM metrics in Prometheus format:
- GPU utilization, power, temperature
- Memory usage (used/free/total)
- Clock frequencies, energy consumption
- Violations and XID errors

### Prometheus Metrics

Multiple endpoints simulate different LLM backends:

| Endpoint | Backend |
|----------|---------|
| `GET /metrics` | Mock server metrics |
| `GET /vllm/metrics` | vLLM-compatible |
| `GET /sglang/metrics` | SGLang-compatible |
| `GET /trtllm/metrics` | TensorRT-LLM-compatible |
| `GET /dynamo_frontend/metrics` | NVIDIA Dynamo frontend |
| `GET /dynamo_component/prefill/metrics` | Dynamo prefill worker |
| `GET /dynamo_component/decode/metrics` | Dynamo decode worker |

**Example metrics:**
- Request latency histograms
- Token counts (prompt/completion)
- In-flight request gauges
- TTFT and ITL distributions

## Behavior Details

### Latency Models

**LLM Endpoints** (`/v1/chat/completions`, `/v1/completions`, `/v1/custom-multimodal`):
- Streaming: TTFT delay before first token, ITL between subsequent tokens
- Non-streaming: TTFT + (ITL × token_count)

**Embeddings**: `base_latency + (per_input_latency × num_inputs)`

**Rankings**: `base_latency + (per_passage_latency × num_passages)`

**TGI Endpoints**: Uses TTFT/ITL model like LLM endpoints

**Fast Mode** (`--fast`): Sets all latencies to zero.

### Token Generation

- Character-based tokenizer (~4 chars/token)
- Output tokens generated by cycling through input tokens
- Deterministic: identical inputs produce identical outputs

| Scenario | Behavior | Finish Reason |
|----------|----------|---------------|
| No `max_tokens` | 0.8-1.2× prompt length (min 16) | `stop` |
| `max_tokens` set | Up to limit, respects `min_tokens` | `length` or `stop` |
| `ignore_eos=true` | Exactly `max_tokens` | `length` |

### Reasoning Models

Models with `gpt-oss` or `qwen` in the name support reasoning:
- Generates `reasoning_content` before main output
- Tokens count toward `max_tokens` budget
- Effort: `low`=100, `medium`=250 (default), `high`=500 tokens

### Error Injection

When `--error-rate` is set, requests randomly fail with HTTP 500. Use `--random-seed` for reproducible error sequences.

## Project Structure

```
tests/aiperf_mock_server/
├── __main__.py      # CLI entry point
├── app.py           # FastAPI application and endpoints
├── config.py        # Configuration (CLI, env vars)
├── models.py        # Pydantic request/response models
├── tokens.py        # Tokenization and generation
├── utils.py         # Request context, streaming, latency
├── metrics.py       # Prometheus metric definitions
├── metrics_utils.py # Metric recording helpers
├── dcgm_faker.py    # GPU telemetry simulation
└── README.md
```

## Examples

### Streaming with Usage

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Explain machine learning"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

### Reasoning Model

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Solve: What is 2+2?"}],
    "reasoning_effort": "high",
    "max_completion_tokens": 1000
  }'
```

### Custom Latency

```bash
# Slow embeddings
aiperf-mock-server --embedding-base-latency 50 --embedding-per-input-latency 10

# Fast rankings
aiperf-mock-server --ranking-base-latency 5 --ranking-per-passage-latency 0.5
```

### Error Injection

```bash
aiperf-mock-server --error-rate 10 --random-seed 42
```

### Multi-GPU Simulation

```bash
# 8 A100 GPUs with auto-scaling load
aiperf-mock-server --dcgm-num-gpus 8 --dcgm-gpu-name a100
```
