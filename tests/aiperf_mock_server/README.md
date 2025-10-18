<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Mock Server

A production-grade OpenAI-compatible mock server for integration testing and performance benchmarking. Features deterministic responses, precise latency simulation, realistic DCGM GPU metrics, and advanced reasoning model support.

## Features

- **OpenAI API Compatibility**: Full support for `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and `/v1/ranking` endpoints
- **Realistic GPU Telemetry**: Dual DCGM metrics endpoints (`/dcgm1/metrics`, `/dcgm2/metrics`) with Prometheus-format GPU metrics (defaults to 2× H200 GPUs)
- **Precise Latency Control**: Configurable Time-to-First-Token (TTFT) and Inter-Token Latency (ITL) for realistic timing simulation
- **Reasoning Model Support**: Native emulation of GPT-OSS/Qwen-style reasoning with `reasoning_content` and configurable `reasoning_effort` levels
- **Error Injection Framework**: Reproducible fault testing with configurable error rates and deterministic seeding
- **High-Throughput Architecture**: Multi-worker support via Uvicorn for concurrent load testing
- **Deterministic Behavior**: Hash-based token generation ensures identical outputs for identical inputs

## Installation

### From Project Root

```bash
make install-mock-server
```

### Standalone Installation

```bash
cd tests/aiperf_mock_server
pip install -e ".[dev]"
```

After installation, the `aiperf-mock-server` command will be available in your environment.

## Quick Start

```bash
# Start with defaults (127.0.0.1:8000, 20ms TTFT, 5ms ITL, 2× H200 GPUs)
aiperf-mock-server

# Custom server configuration
aiperf-mock-server --port 8080 --ttft 50 --itl 10 --workers 4

# Configure GPU simulation (4× H100 GPUs)
aiperf-mock-server --dcgm-num-gpus 4 --dcgm-gpu-name h100

# Enable fault injection for reliability testing
aiperf-mock-server --error-rate 5 --random-seed 42

# Enable verbose logging for debugging
aiperf-mock-server -v
```

## Configuration

Configuration can be provided via CLI arguments or environment variables with `MOCK_SERVER_` prefix.

### Server Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--port` / `-p` | `8000` | Server port (1-65535) |
| `--host` | `127.0.0.1` | Server bind address |
| `--workers` / `-w` | `1` | Uvicorn worker processes (1-32) |
| `--ttft` / `-t` | `20.0` | Time to first token in milliseconds |
| `--itl` | `5.0` | Inter-token latency in milliseconds |
| `--error-rate` | `0.0` | Error injection rate (0-100%) |
| `--random-seed` | `None` | Random seed for reproducible error injection |
| `--log-level` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `--verbose` / `-v` | `false` | Enable DEBUG logging (overrides log-level) |
| `--access-logs` | `false` | Enable HTTP access logs |

### GPU Telemetry Configuration

DCGM metrics are always enabled and available at `/dcgm1/metrics` and `/dcgm2/metrics`.

| Option | Default | Description |
|--------|---------|-------------|
| `--dcgm-gpu-name` | `h200` | GPU model: `rtx6000`, `a100`, `h100`, `h100-sxm`, `h200`, `b200`, `gb200` |
| `--dcgm-num-gpus` | `2` | Number of GPUs to simulate (1-8) |
| `--dcgm-initial-load` | `0.7` | Initial GPU load level (0.0=idle, 1.0=maximum) |
| `--dcgm-hostname` | `localhost` | Hostname label in Prometheus metrics |
| `--dcgm-seed` | `None` | Random seed for deterministic GPU metrics |

### Environment Variables

All configuration options can be set via environment variables:

```bash
export MOCK_SERVER_PORT=8080
export MOCK_SERVER_TTFT=30
export MOCK_SERVER_DCGM_GPU_NAME=h100
aiperf-mock-server
```

## API Endpoints

### Chat Completions

**`POST /v1/chat/completions`**

OpenAI-compatible chat completions with streaming and non-streaming support. Supports reasoning models (models containing `gpt-oss` or `qwen`) with `reasoning_content` field and configurable `reasoning_effort`.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
    "max_completion_tokens": 100,
    "reasoning_effort": "high",
    "stream": false
  }'
```

**Supported Parameters:**
- `model`: Model identifier (string)
- `messages`: Array of message objects with `role` and `content`
- `max_completion_tokens` or `max_tokens`: Maximum output tokens
- `reasoning_effort`: `"low"` (100 tokens) | `"medium"` (250 tokens) | `"high"` (500 tokens)
- `stream`: Enable Server-Sent Events streaming (boolean)
- `stream_options`: Include usage stats in stream with `{"include_usage": true}`
- `min_tokens`: Minimum output tokens to generate
- `ignore_eos`: Generate exactly `max_tokens` tokens

### Text Completions

**`POST /v1/completions`**

Text completion endpoint with support for single or batched prompts. Supports streaming via Server-Sent Events.

**Supported Parameters:**
- `prompt`: String or array of strings
- `max_tokens`: Maximum output tokens
- `stream`: Enable streaming
- `stream_options`, `min_tokens`, `ignore_eos`: Same as chat completions

### Embeddings

**`POST /v1/embeddings`**

Generates deterministic 768-dimensional embeddings based on input text hash. Suitable for testing embedding-based workflows.

**Supported Parameters:**
- `input`: String or array of strings
- `model`: Model identifier

### Rankings

**`POST /v1/ranking`**

Returns deterministic relevance scores for query-passage pairs, sorted by relevance score.

**Supported Parameters:**
- `query`: Object with `text` field
- `passages`: Array of objects with `text` field
- `model`: Model identifier

### Monitoring & Health

**`GET /health`**

Health check endpoint returning server status and current configuration.

**`GET /`**

Server information including version and configuration.

### GPU Telemetry

**`GET /dcgm1/metrics`** and **`GET /dcgm2/metrics`**

DCGM metrics in Prometheus exposition format. Both endpoints provide independent metric streams simulating separate DCGM instances. Metrics are dynamically generated and include:

- GPU utilization percentage
- Power usage and limits (Watts)
- Temperature (GPU and memory, Celsius)
- Memory usage (used/free/total in MiB)
- Clock frequencies (SM and memory, MHz)
- Energy consumption (millijoules)
- Violations (power and thermal, microseconds)
- XID error counts

**Example:**
```bash
# Query first DCGM instance
curl http://localhost:8000/dcgm1/metrics

# Query second DCGM instance
curl http://localhost:8000/dcgm2/metrics
```

## Architecture & Implementation

### Tokenization

Character-based tokenizer approximating ~4 characters per token. Output tokens are deterministically generated by cycling through input tokens, ensuring reproducible responses for identical requests.

### Latency Simulation

- **Streaming Mode**: TTFT delay before first token, then ITL delay between each subsequent token
- **Non-Streaming Mode**: Single delay = TTFT + (ITL × total_tokens)

Timing is precise down to the millisecond using `asyncio.sleep()` with `perf_counter()` for accurate measurements.

### Output Generation Logic

| Scenario | Behavior | Finish Reason |
|----------|----------|---------------|
| No `max_tokens` set | Generates 0.8-1.2× prompt length (minimum 16 tokens) | `stop` |
| `max_tokens` set | Generates up to limit, respects `min_tokens` if specified | `length` if limit reached, else `stop` |
| `ignore_eos=true` | Generates exactly `max_tokens` tokens | `length` |

### Reasoning Model Behavior

Models containing `gpt-oss` or `qwen` in their name automatically support reasoning:

- Generates `reasoning_content` field before primary output
- Reasoning tokens count toward total `max_tokens` budget
- Effort mapping: `low`=100 tokens | `medium`=250 tokens (default) | `high`=500 tokens
- In streaming mode, reasoning tokens are emitted before content tokens

### Error Injection

When `--error-rate` is set, requests randomly fail with HTTP 500 status. The `--random-seed` parameter enables deterministic error sequences for reproducible testing.

### GPU Telemetry Simulation

DCGM metrics are always enabled with two independent endpoints (`/dcgm1/metrics`, `/dcgm2/metrics`):

- **Dynamic Load Simulation**: Metrics evolve with configurable initial load level
- **Per-GPU Variance**: Each GPU has slight random offsets for realistic variation
- **Comprehensive Metrics**: Utilization, power, temperature, memory, clocks, energy, violations, errors
- **Multiple GPU Types**: RTX 6000, A100, H100, H100-SXM, H200, B200, GB200 with model-specific characteristics
- **Prometheus Format**: Standard exposition format compatible with Prometheus, Grafana, and DCGM tooling

## Usage Examples

### Streaming Chat with Usage Statistics

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

### Reasoning Model with High Effort

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

### High-Concurrency Load Testing

```bash
# Start server with 32 workers for maximum throughput
aiperf-mock-server --workers 32 --port 8000
```

### Custom GPU Configuration

```bash
# Simulate 8× A100 GPUs at 50% load
aiperf-mock-server \
  --dcgm-num-gpus 8 \
  --dcgm-gpu-name a100 \
  --dcgm-initial-load 0.5

# Query both DCGM instances
curl http://localhost:8000/dcgm1/metrics
curl http://localhost:8000/dcgm2/metrics
```

### Fault Injection Testing

```bash
# 10% error rate with deterministic seed
aiperf-mock-server --error-rate 10 --random-seed 42

# Test error handling
for i in {1..100}; do
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}'
done
```
