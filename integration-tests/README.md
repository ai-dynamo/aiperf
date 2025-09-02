<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Integration Test Server

A FastAPI server that implements an OpenAI-compatible chat completions API for integration testing.

The server echoes user prompts back token by token with configurable latencies, using the actual tokenizer from the requested model.

## Features

- **OpenAI-compatible API**: Implements `/v1/chat/completions` endpoint
- **Token-by-token echoing**: Uses the actual tokenizer from the requested model
- **Streaming and non-streaming**: Supports both `stream: true` and `stream: false`
- **Configurable latencies**:
  - Time to first token latency (TTFT)
  - Inter-token latency (ITL)
- **Precise timing**: Uses `perf_counter` for accurate latency simulation
- **Flexible configuration**: Environment variables and command-line arguments with unified config system
- **Model-specific tokenization**: Automatically loads tokenizers for different models
- **Pre-loading support**: Pre-load tokenizers at startup for faster responses
- **Runtime configuration**: Configure server settings via `/configure` endpoint
- **Comprehensive logging**: Configurable log levels and access logs
- **Error simulation**: Returns 404 for unsupported models to simulate real-world scenarios

## Installation

```bash
cd integration-tests
pip install -e .
```

Or with development dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

> [!IMPORTANT]
>You must provide a model tokenizer to load either on start or dynamically
> after running using the /configure endpoint, or all requests will return HTTP 404.

### Command Line

```bash
# Basic usage
aiperf-mock-server -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Custom configuration with short flags
aiperf-mock-server -p 8080 -t 30 -i 10 -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Full configuration with long flags
aiperf-mock-server \
  --port 8080 \
  --ttft 30 \
  --itl 10 \
  --host 127.0.0.1 \
  --workers 4 \
  --log-level DEBUG \
  --tokenizer-models gpt2 \
  --tokenizer-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# With environment variables
export MOCK_SERVER_PORT=8080
export MOCK_SERVER_TTFT=30
export MOCK_SERVER_ITL=10
export MOCK_SERVER_LOG_LEVEL=DEBUG
export MOCK_SERVER_TOKENIZER_MODELS='["gpt2", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]'
aiperf-mock-server
```

### Environment Variables

All configuration options can be set via environment variables with the `MOCK_SERVER_` prefix:

- `MOCK_SERVER_PORT`: Port to run the server on (default: 8000)
- `MOCK_SERVER_HOST`: Host to bind to (default: 0.0.0.0)
- `MOCK_SERVER_WORKERS`: Number of uvicorn worker processes (default: 1)
- `MOCK_SERVER_TTFT`: Time to first token latency in milliseconds (default: 20.0)
- `MOCK_SERVER_ITL`: Inter-token latency in milliseconds (default: 5.0)
- `MOCK_SERVER_LOG_LEVEL`: Logging level (default: INFO)
- `MOCK_SERVER_ACCESS_LOGS`: Enable HTTP access logs (default: false)
- `MOCK_SERVER_TOKENIZER_MODELS`: JSON-formatted array of models to pre-load

### API Usage

#### Non-streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "max_tokens": 10,
    "stream": false
  }'
```

#### Streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "max_tokens": 10,
    "stream": true
  }'
```

#### Runtime Configuration

```bash
# Configure latencies at runtime
curl -X POST http://localhost:8000/configure \
  -H "Content-Type: application/json" \
  -d '{
    "ttft": 100,
    "itl": 25,
    "tokenizer_models": ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Server Information

```bash
curl http://localhost:8000/
```

## Configuration Options

| Parameter | CLI Flag | Environment Variable | Default | Description |
|-----------|----------|---------------------|---------|-------------|
| Port | `--port`, `-p` | `MOCK_SERVER_PORT` | 8000 | Server port |
| Host | `--host`, `-h` | `MOCK_SERVER_HOST` | 0.0.0.0 | Server host |
| Workers | `--workers`, `-w` | `MOCK_SERVER_WORKERS` | 1 | Worker processes for uvicorn server |
| TTFT | `--ttft`, `-t` | `MOCK_SERVER_TTFT` | 20.0 | Time to first token (ms) |
| ITL | `--itl`, `-i` | `MOCK_SERVER_ITL` | 5.0 | Inter-token latency (ms) |
| Log Level | `--log-level`, `-l` | `MOCK_SERVER_LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| Access Logs | `--access-logs`, `-a` | `MOCK_SERVER_ACCESS_LOGS` | false | Enable HTTP access logs |
| Tokenizer Models | `--tokenizer-models`, `-m` | `MOCK_SERVER_TOKENIZER_MODELS` | [] | Models to pre-load at startup |

Configuration priority (highest to lowest):
1. CLI arguments
2. Environment variables (prefixed with `MOCK_SERVER_`)
3. Default values

## API Endpoints

### POST `/v1/chat/completions`
OpenAI-compatible chat completions endpoint that echoes user messages token by token.

**Request Body**: Standard OpenAI chat completions format
**Response**: OpenAI-compatible response (streaming or non-streaming)

### POST `/configure`
Runtime configuration endpoint for updating server settings.

**Request Body**:
```json
{
  "ttft": 50,
  "itl": 15,
  "tokenizer_models": ["gpt2", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
}
```

### GET `/health`
Health check endpoint returning server status and current configuration.

### GET `/`
Root endpoint providing server information and available endpoints.

## How It Works

1. **Request Processing**: The server receives a chat completion request
2. **Tokenization**: Uses the model-specific tokenizer to tokenize the user prompt
3. **Token Limit**: Respects the `max_tokens` parameter if specified
4. **Latency Simulation**:
   - Waits for the configured TTFT before sending the first token
   - Waits for the configured ITL between subsequent tokens
   - Uses `perf_counter` for precise timing control
5. **Response**: Echoes back the tokenized prompt either as:
   - A complete response (non-streaming)
   - Token-by-token chunks (streaming)

## Supported Models

The server uses Hugging Face Transformers to load tokenizers for any supported model. Models must be:
- Available on Hugging Face Hub
- Compatible with `AutoTokenizer.from_pretrained()`

If a tokenizer fails to load for a requested model, the server returns a 404 error to simulate model unavailability.

## Development

### Code Structure

```
mock_server/
├── __init__.py          # Package initialization
├── app.py               # FastAPI application with endpoints
├── config.py            # Unified configuration management (CLI + env vars)
├── main.py              # CLI entry point with cyclopts
├── models.py            # Pydantic models for API
└── tokenizer_service.py # Tokenizer management and caching
```
