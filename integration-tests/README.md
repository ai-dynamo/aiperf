<!--
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Mock Integration Test Server

A FastAPI server that implements OpenAI-compatible chat completions API for integration testing. The server echoes user prompts back token by token with configurable latencies, using the actual tokenizer from the requested model.

## Features

- **OpenAI-compatible API**: Implements `/v1/chat/completions` endpoint
- **Token-by-token echoing**: Uses the actual tokenizer from the requested model
- **Streaming and non-streaming**: Supports both `stream: true` and `stream: false`
- **Configurable latencies**:
  - Time to first token latency (TTFT)
  - Inter-token latency (ITL)
- **Precise timing**: Uses `perf_counter` for accurate latency simulation
- **Flexible configuration**: Environment variables and command-line arguments
- **Model-specific tokenization**: Automatically loads tokenizers for different models
- **Runtime configuration**: Dynamic configuration updates via `/configure` endpoint

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

### Command Line

```bash
# Basic usage
aiperf-mock-server

# Custom configuration
aiperf-mock-server \
  --port 8080 \
  --ttft 30 \
  --itl 10 \
  --host 127.0.0.1 \
  --workers 2 \
  --log-level DEBUG

# With environment variables
export MOCK_SERVER_PORT=8080
export MOCK_SERVER_TTFT_MS=30
export MOCK_SERVER_ITL_MS=10
export MOCK_SERVER_WORKERS=2
aiperf-mock-server
```

### Environment Variables

- `MOCK_SERVER_PORT`: Port to run the server on (default: 8000)
- `MOCK_SERVER_HOST`: Host to bind to (default: 0.0.0.0)
- `MOCK_SERVER_TTFT_MS`: Time to first token latency in milliseconds (default: 50.0)
- `MOCK_SERVER_ITL_MS`: Inter-token latency in milliseconds (default: 10.0)
- `MOCK_SERVER_WORKERS`: Number of uvicorn worker processes (default: 1)

### API Usage

#### Non-streaming Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "max_tokens": 10,
    "stream": false
  }'
```

#### Streaming Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "max_tokens": 10,
    "stream": true
  }'
```

#### Runtime Configuration

```bash
# Update latency settings dynamically
curl -X POST http://localhost:8000/configure \
  -H "Content-Type: application/json" \
  -d '{
    "ttft_ms": 100,
    "itl_ms": 25
  }'
```

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Server Information

```bash
curl http://localhost:8000/
```

## Configuration Options

| Parameter | CLI Flag | CLI Alias | Environment Variable | Default | Description |
|-----------|----------|-----------|---------------------|---------|-------------|
| Port | `--port` | `-p` | `MOCK_SERVER_PORT` | 8000 | Server port |
| Host | `--host` | `-h` | `MOCK_SERVER_HOST` | 0.0.0.0 | Server host |
| TTFT | `--ttft` | `--time-to-first-token-ms` | `MOCK_SERVER_TTFT_MS` | 50.0 | Time to first token (ms) |
| ITL | `--itl` | `--inter-token-latency-ms` | `MOCK_SERVER_ITL_MS` | 10.0 | Inter-token latency (ms) |
| Workers | `--workers` | `-w` | `MOCK_SERVER_WORKERS` | 1 | Worker processes for uvicorn server |
| Log Level | `--log-level` | | | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server information and current configuration |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/health` | GET | Health check endpoint |
| `/configure` | POST | Runtime configuration updates |

## How It Works

1. **Request Processing**: The server receives a chat completion request
2. **Tokenization**: Uses the model-specific tokenizer to tokenize the user prompt
3. **Token Limit**: Respects the `max_tokens` parameter if specified
4. **Latency Simulation**:
   - Waits for the configured TTFT before sending the first token
   - Waits for the configured ITL between subsequent tokens
5. **Response**: Echoes back the tokenized prompt either as:
   - A complete response (non-streaming)
   - Token-by-token chunks (streaming)

## Supported Models

The server automatically loads tokenizers for any model supported by Hugging Face Transformers. If a tokenizer fails to load, it falls back to GPT-2.

## Development

### Running Tests

```bash
pytest
```

### Code Structure

```
mock_server/
├── __init__.py          # Package initialization
├── app.py               # FastAPI application
├── config.py            # Configuration management
├── main.py              # CLI entry point
├── models.py            # Pydantic models
└── tokenizer_service.py # Tokenizer management
```

## Dependencies

- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **Transformers**: Hugging Face tokenizers
- **Pydantic**: Data validation and settings management
- **Typer**: Command-line interface framework
