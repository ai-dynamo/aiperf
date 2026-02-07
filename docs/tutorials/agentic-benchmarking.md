<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic Benchmarking (Agentic Coding)

Benchmark LLM inference servers using pre-recorded agentic trajectories that replay real-world multi-turn tool-calling conversations.

## Overview

The Agentic Coding dataset format captures complete agent trajectories — multi-turn conversations where an LLM interacts with tools to accomplish a task. Each trajectory records the cumulative message history at every step, including system prompts, user instructions, assistant responses, tool calls, and tool results.

This is fundamentally different from synthetic multi-turn benchmarks:

| Aspect | Synthetic Multi-Turn | Agentic Coding |
|--------|---------------------|---------|
| **Prompts** | Generated from token distributions | Real agent conversations |
| **Tool calls** | Not supported | Full tool call/response history |
| **Turn count** | Configurable mean/stddev | Fixed per trajectory |
| **Message content** | Synthetic text | Pre-recorded cumulative history |
| **Output length** | Configurable or random | Estimated from original trajectory |
| **Inter-turn delay** | Configurable | Zero (agent execution) |

**Key behaviors:**

- **Sequential processing**: Trajectories are consumed in order — each one is a complete agent session
- **Response discarding**: The LLM's response is discarded after each turn because the next turn already contains the pre-recorded assistant response from the original trajectory
- **Zero delay**: Turns execute back-to-back with no inter-turn delay, matching real agent execution
- **Delta de-duplication**: Cumulative message history is stored as deltas to reduce memory from O(N^2) to O(N)

## Data Format

Agentic Coding datasets are JSONL files where each line is a JSON object with these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conversation_id` | `string` | Yes | Trajectory identifier (groups related entries) |
| `conversation_idx` | `integer` | Yes | Step index within the trajectory (0-based, sequential, no gaps) |
| `messages` | `list[dict]` | Yes | Cumulative OpenAI-format message history (min 1 message) |
| `tools` | `list[dict]` | No | Tool/function definitions for this conversation |
| `type` | `string` | No | Optional explicit type field (`"agentic_coding"`) |

### Cumulative Message Format

Each entry's `messages` field contains the **full** message history up to that step. The loader automatically computes deltas (new messages per step) to avoid storing redundant data.

### Example Dataset

A 3-step trajectory where an agent lists files and reads one:

```jsonl
{"conversation_id": "traj-001", "conversation_idx": 0, "messages": [{"role": "system", "content": "You are a coding assistant with file access."}, {"role": "user", "content": "What files are in the src/ directory?"}], "tools": [{"type": "function", "function": {"name": "list_files", "description": "List files in a directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]}
{"conversation_id": "traj-001", "conversation_idx": 1, "messages": [{"role": "system", "content": "You are a coding assistant with file access."}, {"role": "user", "content": "What files are in the src/ directory?"}, {"role": "assistant", "content": "I'll check the directory for you.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "list_files", "arguments": "{\"path\": \"src/\"}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "main.py\nutils.py\nconfig.py"}], "tools": [{"type": "function", "function": {"name": "list_files", "description": "List files in a directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]}
{"conversation_id": "traj-001", "conversation_idx": 2, "messages": [{"role": "system", "content": "You are a coding assistant with file access."}, {"role": "user", "content": "What files are in the src/ directory?"}, {"role": "assistant", "content": "I'll check the directory for you.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "list_files", "arguments": "{\"path\": \"src/\"}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "main.py\nutils.py\nconfig.py"}, {"role": "assistant", "content": "The src/ directory contains three files:\n- main.py\n- utils.py\n- config.py\n\nWould you like me to read any of them?"}], "tools": [{"type": "function", "function": {"name": "list_files", "description": "List files in a directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]}
```

**What happens at each step:**

| Step | Delta (new messages) | What the LLM sees |
|------|---------------------|-------------------|
| 0 | system + user | Initial prompt — LLM generates first response |
| 1 | assistant + tool_call + tool_result | Full history including pre-recorded response and tool result |
| 2 | assistant | Full history — LLM generates final response |

A single file can contain multiple trajectories (different `conversation_id` values). They are grouped and processed independently.

### Validation Rules

The loader enforces these rules during `load_dataset()`:

- Each `conversation_id` group must have `conversation_idx` values starting at 0 with no gaps
- Duplicate indices within a group raise a `ValueError`
- Empty lines in the JSONL file are skipped
- `messages` must contain at least one message
- `conversation_idx` must be non-negative

## Quick Start

### Setting Up the Server

```bash
# Start a vLLM server with tool calling support
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

### Running the Benchmark

```bash
# Basic Agentic Coding benchmark
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --input-file trajectories.jsonl \
    --custom-dataset-type agentic_coding \
    --concurrency 4
```

The dataset type can be auto-detected from the file contents, so `--custom-dataset-type agentic_coding` is optional but recommended for clarity.

## How It Works

### Data Flow

```
JSONL File
  │
  ▼
AgenticCodingDatasetLoader.load_dataset()
  ├── Parse entries, group by conversation_id
  ├── Sort each group by conversation_idx
  └── Validate sequential indexing (no gaps/duplicates)
  │
  ▼
AgenticCodingDatasetLoader.convert_to_conversations()
  ├── Compute delta messages (only new messages per step)
  ├── Skip duplicate entries (empty deltas)
  └── Create Conversation with discard_assistant_response=True
  │
  ▼
Worker execution
  ├── ChatEndpoint concatenates raw_messages from all turns
  ├── Sends full message history to LLM API
  ├── Receives response (discarded — next turn has pre-recorded history)
  └── Advances to next turn
```

### Delta De-duplication

The raw Agentic Coding format stores cumulative message history per entry. Storing all entries as-is would use O(N^2) memory for an N-turn trajectory. Instead, the loader computes deltas — only the new messages per step:

```
Entry 0 messages: [system, user]           → Delta: [system, user]
Entry 1 messages: [system, user, asst, tool] → Delta: [asst, tool]
Entry 2 messages: [system, user, asst, tool, asst] → Delta: [asst]
```

At request time, the `ChatEndpoint` reconstructs the full history by concatenating `raw_messages` from all turns in the `turn_list`:

```python
# ChatEndpoint builds the full message list from turn deltas
messages = [
    msg
    for turn in turns
    if turn.raw_messages is not None
    for msg in turn.raw_messages
]
```

### Response Discarding

Agentic Coding conversations set `discard_assistant_response=True`. This tells the worker to **not** store the LLM's response in the session's `turn_list`. The next turn's delta already contains the original assistant response from the recorded trajectory.

Without discarding, the session would accumulate both the LLM's response AND the pre-recorded response, resulting in duplicated messages in the context.

### Duplicate Entry Handling

Some datasets contain consecutive entries with identical message counts (recording artifacts). These produce empty deltas and are automatically skipped — no turn is created, no request is sent to the LLM.

## Advanced Usage

### Pre-computed ISL with `--tokenizer`

By default, ISL (input sequence length) for agentic coding turns is computed post-hoc at record processing time using basic token counting — this misses chat template overhead (role markers, special tokens, tool formatting).

When you provide `--tokenizer`, AIPerf pre-computes accurate ISL at dataset load time using `apply_chat_template` on the cumulative messages for each turn, matching what the LLM server actually tokenizes:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --input-file trajectories.jsonl \
    --custom-dataset-type agentic_coding \
    --tokenizer Qwen/Qwen3-0.6B \
    --concurrency 4
```

This is recommended for accurate ISL reporting, especially when tool definitions add significant token overhead.

### Streaming Mode

Agentic Coding works with both streaming and non-streaming endpoints:

```bash
# With streaming
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --input-file trajectories.jsonl \
    --custom-dataset-type agentic_coding \
    --streaming \
    --concurrency 4
```

### Multiple Server Instances

Load balance across multiple servers:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url server1:8000 \
    --url server2:8000 \
    --input-file trajectories.jsonl \
    --custom-dataset-type agentic_coding \
    --concurrency 8
```

### Controlling Request Count

By default, AIPerf processes all trajectories in the file. Use `--request-count` to limit the total number of individual turn requests sent:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --input-file trajectories.jsonl \
    --custom-dataset-type agentic_coding \
    --request-count 100 \
    --concurrency 4
```

## Creating Agentic Coding Datasets

### From Agent Frameworks

Agentic Coding datasets are typically generated by recording agent execution traces. The key requirement is that each trajectory step contains the full cumulative message history in OpenAI chat format.

Frameworks that produce compatible traces:
- **OpenCode**: Records multi-model agent trajectories with tool calls
- **Custom harnesses**: Any system that logs OpenAI-format message history per step

### Manual Construction

For testing, you can construct entries manually. The critical rules:

1. **Cumulative history**: Each entry's `messages` must include all messages from prior entries plus new ones
2. **Sequential indexing**: `conversation_idx` must be 0, 1, 2, ... with no gaps
3. **Consistent tools**: The `tools` field should be the same across all entries in a trajectory (the loader uses the first entry's tools)
4. **OpenAI message format**: Messages must follow the OpenAI chat completions format with `role` and `content` fields

### Validating Your Dataset

Use auto-detection to verify your file is recognized:

```bash
# Auto-detection will confirm the format
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --input-file your_dataset.jsonl \
    --endpoint-type chat \
    --concurrency 1 \
    --request-count 1
```

If the file is not recognized, you'll see: `"No loader can handle the data format. Please specify --custom-dataset-type explicitly."` Check that your entries have the required `conversation_id`, `conversation_idx`, and `messages` fields.

## Quick Reference

**Required CLI flags:**
- `--input-file <path>` — Path to Agentic Coding JSONL file
- `--model <name>` — Model to benchmark
- `--url <url>` — API server URL
- `--endpoint-type chat` — Must use the chat endpoint type

**Recommended CLI flags:**
- `--custom-dataset-type agentic_coding` — Explicit format (auto-detected if omitted)
- `--tokenizer <name>` — Pre-compute accurate ISL using `apply_chat_template`

**Sampling strategy:**
- Defaults to `sequential` — trajectories are consumed in order
- Override with `--dataset-sampling-strategy` if needed

**Key behaviors:**
- Turns have zero inter-turn delay (agent execution pattern)
- LLM responses are discarded (next turn has pre-recorded history)
- Duplicate entries (recording artifacts) are automatically skipped
- Tool definitions are taken from the first entry in each trajectory
