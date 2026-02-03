<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting Started with AIPerf

This guide will help you install AIPerf and run your first benchmark in minutes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Install via pip](#install-via-pip)
  - [Install from Source](#install-from-source)
  - [Using Docker](#using-docker)
- [Running Your First Benchmark](#running-your-first-benchmark)
  - [Step 1: Start an Inference Server](#step-1-start-an-inference-server)
  - [Step 2: Run AIPerf](#step-2-run-aiperf)
  - [Step 3: Understanding the Output](#step-3-understanding-the-output)
- [Common First-Run Issues](#common-first-run-issues)
- [Next Steps](#next-steps)

## Prerequisites

Before installing AIPerf, ensure you have:

**Required:**
- Python 3.10 or higher
- pip package manager
- A target LLM inference server to benchmark (e.g., vLLM, TGI, Triton, etc.)

**Recommended:**
- Linux or macOS (Windows via WSL2)
- 4+ CPU cores for multi-worker operation
- 8GB+ RAM for moderate workloads
- Network access to your inference server

**For GPU Telemetry (Optional):**
- NVIDIA GPU with DCGM Exporter running
- See [GPU Telemetry Tutorial](tutorials/gpu-telemetry.md)

## Installation

### Install via pip

The simplest installation method:

```bash
pip install aiperf
```

Verify installation:

```bash
aiperf --version
```

### Install from Source

For development or the latest features:

```bash
# Clone the repository
git clone https://github.com/ai-dynamo/aiperf.git
cd aiperf

# Install using uv (recommended)
make first-time-setup

# Or using pip
pip install -e .
```

Verify installation:

```bash
aiperf --version
```

### Using Docker

Run AIPerf in a containerized environment:

```bash
# Pull the latest image
docker pull aidynamo/aiperf:latest

# Run a benchmark
docker run --rm --network=host \
  aidynamo/aiperf:latest \
  profile \
    --model your_model \
    --url http://your-server:8000 \
    --endpoint-type chat \
    --streaming \
    --request-count 100
```

See [Deployment Guide](deployment.md) for advanced Docker and Kubernetes setups.

## Running Your First Benchmark

Let's run a simple benchmark against a local vLLM server.

### Step 1: Start an Inference Server

If you don't have an inference server running, start vLLM with a small model:

```bash
# Pull and run vLLM Docker container
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000
```

Wait for the server to be ready (usually 30-60 seconds). You'll see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Run AIPerf

Open a new terminal and run your first benchmark:

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 10 \
  --request-count 100
```

**Command Breakdown:**
- `--model`: Name of the model being benchmarked
- `--url`: Base URL of your inference server
- `--endpoint-type chat`: Use OpenAI-compatible chat completions API
- `--streaming`: Enable streaming responses to measure TTFT and ITL
- `--concurrency 10`: Maintain 10 concurrent requests
- `--request-count 100`: Send a total of 100 requests

You'll see a real-time dashboard showing progress:

```
NVIDIA AIPerf | LLM Metrics
┌─────────────────────────────┬─────────┬────────┬────────┬────────┐
│ Metric                      │     avg │    min │    max │    p99 │
├─────────────────────────────┼─────────┼────────┼────────┼────────┤
│ Time to First Token (ms)    │   18.26 │  11.22 │ 106.32 │  68.82 │
│ Inter Token Latency (ms)    │   11.23 │   8.80 │  13.17 │  12.48 │
│ Request Latency (ms)        │  487.30 │ 267.07 │ 769.57 │ 715.99 │
│ Output Token Throughput     │ 10944.0 │    N/A │    N/A │    N/A │
│ Request Throughput (req/s)  │  255.54 │    N/A │    N/A │    N/A │
└─────────────────────────────┴─────────┴────────┴────────┴────────┘
```

The benchmark typically completes in 10-30 seconds depending on your hardware.

### Step 3: Understanding the Output

AIPerf generates several output files in the `artifacts/` directory:

**Summary Files:**
```
artifacts/Qwen_Qwen3-0.6B-chat-concurrency10/
├── profile_export_aiperf.csv      # Metrics in CSV format
├── profile_export_aiperf.json     # Metrics with metadata
├── profile_export.jsonl           # Per-request metrics
└── profile_export_raw.jsonl       # Raw request/response data
```

**Key Metrics Explained:**

| Metric | What It Measures | Good Values |
|--------|------------------|-------------|
| **Time to First Token (TTFT)** | Latency until first token arrives | < 50ms for interactive |
| **Inter Token Latency (ITL)** | Time between tokens during generation | < 20ms for smooth streaming |
| **Request Latency** | Total end-to-end request time | Depends on output length |
| **Output Token Throughput** | Total tokens generated per second | Higher is better |
| **Request Throughput** | Completed requests per second | Higher is better |

For a complete metrics guide, see [Metrics Reference](metrics_reference.md).

## Common First-Run Issues

### Connection Refused

**Error:**
```
ERROR: Failed to connect to http://localhost:8000
```

**Solutions:**
1. Verify your server is running: `curl http://localhost:8000/health`
2. Check the port number matches your server configuration
3. If using Docker, ensure `--network=host` or proper port mapping

### Model Not Found

**Error:**
```
ERROR: Model 'YourModel' not found on server
```

**Solutions:**
1. Verify the model name exactly matches what's loaded on the server
2. Check server logs for the loaded model name
3. Some servers use different naming (with/without organization prefix)

### Out of Memory (OOM)

**Error:**
```
ERROR: CUDA out of memory
```

**Solutions:**
1. Reduce `--concurrency` (try 4, 2, or 1)
2. Use smaller `--prompt-input-tokens-mean` value
3. Reduce `--prompt-output-tokens-mean` value
4. Use a smaller model

### Python Version Error

**Error:**
```
ERROR: Python 3.10 or higher required
```

**Solution:**
```bash
# Check your Python version
python --version

# Use a newer Python version or pyenv
pyenv install 3.11
pyenv local 3.11
```

### Workers Not Starting

**Error:**
```
WARNING: Workers failed to start
```

**Solutions:**
1. Check system resources (CPU, memory)
2. Reduce `--workers-max` value
3. Check logs for specific worker errors
4. Try `--log-level DEBUG` for detailed diagnostics

For more issues, see [Troubleshooting Guide](troubleshooting.md).

## Next Steps

Now that you've run your first benchmark, explore more features:

### Basic Benchmarking
- **[Tutorial](tutorial.md)** - More complete examples
- **[CLI Options](cli_options.md)** - All available command-line options
- **[Examples Gallery](examples.md)** - Real-world benchmark scenarios

### Load Patterns
- **[Request Rate Mode](tutorials/request-rate-concurrency.md)** - Control request timing
- **[Arrival Patterns](tutorials/arrival-patterns.md)** - Realistic traffic simulation
- **[Warmup Phase](tutorials/warmup.md)** - Eliminate cold-start effects

### Analysis
- **[Goodput](tutorials/goodput.md)** - Measure SLO compliance
- **[Timeslices](tutorials/timeslices.md)** - Time-windowed analysis
- **[Plotting](tutorials/plot.md)** - Visualize results

### Advanced Features
- **[GPU Telemetry](tutorials/gpu-telemetry.md)** - Monitor GPU metrics
- **[Custom Datasets](tutorials/custom-prompt-benchmarking.md)** - Use your own data
- **[Plugin System](plugins/plugin-system.md)** - Extend AIPerf

### Production Use
- **[Best Practices](best-practices.md)** - Guidelines for effective benchmarking
- **[Performance Tuning](performance-tuning.md)** - Optimize AIPerf
- **[Deployment Guide](deployment.md)** - Production deployments

## Need Help?

- **Issues**: Report bugs at [GitHub Issues](https://github.com/ai-dynamo/aiperf/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/ai-dynamo/aiperf/discussions)
- **Discord**: Join our community at [Discord](https://discord.gg/D92uqZRjCZ)
- **FAQ**: Check [Frequently Asked Questions](faq.md)
