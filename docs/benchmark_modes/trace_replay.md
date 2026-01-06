<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Trace Replay

This tutorial takes you through an example trace replay profile. Trace Replay benchmarking helps reproduce performance benchmarking results for validation or testing your system under a specific load pattern.

## Start a vLLM Server

Launch a vLLM server with a chat model:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

Verify the server is ready:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

## Mooncake Trace Format

Mooncake provides a specification and sample datasets for [trace replay](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#-open-source-trace) that can be replayed for performance benchmarking.

Mooncake traces use a JSONL file where each line represents a request with timing information.

Required fields for trace replay:
- `timestamp`: Request arrival time in milliseconds
- `input_length`: Number of input tokens
- `output_length`: Number of output tokens
- `hash_ids`: List of block hashes (optional)

Example entry:

```json
{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
```

## Profile using a Custom Trace File

Create a trace file with timing information:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > custom_trace.jsonl << 'EOF'
{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
{"timestamp": 10535, "input_length": 672, "output_length": 26, "hash_ids": [46, 47]}
{"timestamp": 27482, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
EOF
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->
Run AIPerf with the trace file:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file custom_trace.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

The `--fixed-schedule` flag tells AIPerf to send requests at the exact timestamps specified in the trace. This reproduces the original timing pattern.

## Profile using real Mooncake Trace

For real-world benchmarking, use the FAST25 production trace data from the Mooncake research paper:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Download the Mooncake trace data
curl -o mooncake_trace.jsonl https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl

# Create a subset for quick testing
head -n 10 mooncake_trace.jsonl > mooncake_trace_short.jsonl

# Run the trace replay
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file mooncake_trace_short.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->