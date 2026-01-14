<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Trace Replay with Mooncake Traces

This tutorial covers replaying production traces using the Mooncake trace format. Trace replay benchmarking reproduces real-world traffic patterns with precise timing control, enabling performance validation and capacity planning under realistic load.

## When to Use This Tutorial

Use this approach when you need to:
- Replay production traffic patterns captured from real systems
- Validate performance with industry-standard Mooncake FAST'25 traces
- Test system behavior under specific temporal load patterns
- Reproduce benchmark results for regression testing

For other use cases:
- **Custom prompts without timing**: See [Custom Prompt Benchmarking](../tutorials/custom-prompt-benchmarking.md)
- **Precise timestamp control for any dataset**: See [Fixed Schedule](../tutorials/fixed-schedule.md)
- **Multi-turn conversations from files**: See [Multi-Turn Conversations](../tutorials/multi-turn.md)

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
{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [0, 1, 2]}
```

## Profile using a Custom Trace File

Create a trace file with timing information:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
cat > custom_trace.jsonl << 'EOF'
{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [0, 1, 2]}
{"timestamp": 10535, "input_length": 672, "output_length": 26, "hash_ids": [0, 3, 4, 5]}
{"timestamp": 27482, "input_length": 655, "output_length": 52, "hash_ids": [0, 6, 7]}
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