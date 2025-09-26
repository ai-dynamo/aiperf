<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Trace Benchmarking with Mooncake Traces

Trace benchmarking allows you to send
**exactly the payloads specified in your trace file**.
Unlike other dataset types that sample or generate requests, mooncake traces
with `text_input` provide complete control over what gets sent to your inference server.

## Overview

The `mooncake_trace` dataset type with `text_input` provides:

- **Exact Payload Control**: Send precisely the text you specify
- **Deterministic Testing**: Same trace file produces identical requests every time
- **Production Replay**: Use real user queries for realistic benchmarking
- **Debugging**: Isolate performance issues with specific prompts

This is different from `random_pool` which samples from a dataset.
Traces send each entry exactly once in order.

## Creating Mooncake Trace Files

### Basic Text Input Traces

Create trace files with exact text payloads:

<!-- create-mooncake-trace -->
```bash
# Create a trace file with specific text inputs
cat > production_queries.jsonl << 'EOF'
{"text_input": "What is the capital of France?", "output_length": 20}
{"text_input": "Explain quantum computing in simple terms.", "output_length": 100}
{"text_input": "Write a Python function to calculate fibonacci numbers.", "output_length": 150}
{"text_input": "Summarize the main causes of World War II.", "output_length": 200}
{"text_input": "How do neural networks learn?", "output_length": 80}
EOF
```
<!-- /create-mooncake-trace -->

**Field Descriptions:**
- `text_input`: The exact text prompt to send (required if not
providing input_length instead).
- `output_length`: Maximum tokens to generate in response (optional)

### Setting Up the Server

<!-- setup-vllm-trace-benchmarking -->
```bash
# Start vLLM server
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000
```
<!-- /setup-vllm-trace-benchmarking -->

<!-- health-check-vllm-trace-benchmarking -->
```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```
<!-- /health-check-vllm-trace-benchmarking -->

### Running Basic Trace Benchmark

<!-- aiperf-run-trace -->
```bash
# Run with exact text payloads from trace file
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --input-file production_queries.jsonl \
    --custom-dataset-type mooncake_trace \
    --concurrency 2 \
    --warmup-request-count 1
```
<!-- /aiperf-run-mooncake-trace -->

**Key Points:**
- Each line in the JSONL file becomes exactly one request
- Requests are sent in the order they appear in the file
- The `text_input` is sent exactly as specified


### Use Cases

**Perfect for:**
- Regression testing (detecting performance changes)
- A/B testing different model configurations
- Debugging specific prompt performance
- Production workload replay

**Not ideal for:**
- Load testing with varied request patterns (use `random_pool` instead)
- Scalability testing requiring many unique requests
