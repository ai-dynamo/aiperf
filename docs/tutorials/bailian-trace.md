<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile with Bailian Traces

AIPerf supports benchmarking using the [Bailian usage traces](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon), a public dataset of anonymized production chat traces from Qwen model serving. The dataset contains both single-turn requests and multi-turn conversations.

This guide covers replaying Bailian traces with precise timing to reproduce real-world traffic patterns.

---

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

---

## Bailian Trace Format

Bailian traces are JSONL files where each line represents a single request.

- `chat_id`: Randomized unique chat identifier
- `timestamp`: Request arrival time in seconds (converted to milliseconds internally)
- `input_length`: Input token count
- `output_length`: Output token count
- `parent_chat_id`: Parent chat ID linking turns in a session; `-1` for root (default: `-1`)
- `type`: Request type (`text`, `search`, `image`, `file`)
- `turn`: Conversation turn number (default: `1`)
- `hash_ids`: Salted SipHash block IDs for KV cache simulation (16 tokens per block)

Example entries:

```json
{"chat_id": 159, "parent_chat_id": -1, "timestamp": 61.114, "input_length": 521, "output_length": 132, "type": "text", "turn": 1}
{"chat_id": 160, "parent_chat_id": 159, "timestamp": 62.5, "input_length": 400, "output_length": 80, "type": "text", "turn": 2}
```

Entries with the same root `chat_id` form a session and are replayed in `turn` order.

---

## Download and Profile

Download a trace file from the public Bailian dataset:

The repository includes four traces representing different workload types: `qwen_traceA_blksz_16.jsonl`, `qwen_traceB_blksz_16.jsonl`, `qwen_coder_blksz_16.jsonl`, and `qwen_thinking_blksz_16.jsonl`. Substitute any of them in the command below.

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Download a trace file
curl -Lo qwen_traceA_blksz_16.jsonl \
  https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_traceA_blksz_16.jsonl

# Create a small subset for a quick test
head -n 50 qwen_traceA_blksz_16.jsonl > bailian_short.jsonl

# Run trace replay
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file bailian_short.jsonl \
    --custom-dataset-type bailian_trace \
    --fixed-schedule
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

---

## Related Tutorials

- [Trace Benchmarking with Mooncake](../benchmark-modes/trace-replay.md) - Mooncake FAST'25 trace replay
- [Fixed Schedule](fixed-schedule.md) - Precise timestamp-based execution for any dataset
- [Prefix Synthesis](prefix-synthesis.md) - KV cache testing with hash-based prefix data
- [Multi-Turn Conversations](multi-turn.md) - Multi-turn conversation benchmarking
- [Conversation Context Mode](../reference/conversation-context-mode.md) - How conversation history accumulates in multi-turn
