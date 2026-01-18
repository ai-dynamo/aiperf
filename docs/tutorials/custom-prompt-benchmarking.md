<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Custom Prompt Benchmarking

Benchmark with prompts from your own file, sent exactly as specified without sampling or generation.

## Overview

This tutorial uses the `mooncake_trace` dataset type with `text_input` field to send prompts exactly as-is.

The `mooncake_trace` dataset type with `text_input` provides:

- **Exact Control**: Send precisely the text you specify
- **Deterministic Testing**: Same file produces identical request sequence every time
- **Production Replay**: Use real user queries for realistic benchmarking
- **Debugging**: Isolate performance issues with specific prompts

This is different from `random_pool` which samples from a dataset.
Traces send each entry exactly once in order.

### Setting Up the Server

```bash
# Start vLLM server
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

### Running the Benchmark

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
# Create an input file with specific text inputs
```bash
# Create an input file to use for benchmarking
# Text can be provided via text_input or input_length
# Output_length can optionally set the maximum number of tokens to generate in the response
cat > production_queries.jsonl << 'EOF'
{"text_input": "What is the capital of France?", "output_length": 20}
{"text_input": "Explain quantum computing in simple terms.", "output_length": 100}
{"text_input": "Write a Python function to calculate fibonacci numbers.", "output_length": 150}
{"text_input": "Summarize the main causes of World War II.", "output_length": 200}
{"text_input": "How do neural networks learn?", "output_length": 80}
EOF
# Run with exact text payloads
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
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

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
