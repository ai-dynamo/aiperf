---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Inputs JSON Replay
---

# Inputs JSON Replay

Replay pre-formatted multi-turn API payloads from AIPerf's `inputs.json` file format.

## Overview

Every AIPerf benchmark run produces an `inputs.json` artifact in the output directory. This file captures the exact API request payloads that were sent during the benchmark, organized by session. The `inputs_json` dataset type reads this file back and replays its payloads verbatim.

### When to Use

- **Reproducible replay**: Re-run a previous benchmark with the exact same payloads
- **Cross-server comparison**: Run identical payloads against different inference servers
- **Payload editing**: Modify specific payloads in the JSON file, then replay
- **Debugging**: Isolate specific sessions or turns from a prior run for investigation

---

## File Format

The file is a single JSON object with a top-level `data` array. Each element represents one session with an ordered list of API request payloads.

```json
{
  "data": [
    {
      "session_id": "session-001",
      "payloads": [
        {
          "messages": [{"role": "user", "content": "Hello"}],
          "model": "Qwen/Qwen3-0.6B",
          "max_tokens": 1024
        },
        {
          "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
          ],
          "model": "Qwen/Qwen3-0.6B",
          "max_tokens": 1024
        }
      ]
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data` | array | Yes | Top-level array of session objects |
| `data[].session_id` | string | Yes | Unique identifier for the session |
| `data[].payloads` | array | Yes (non-empty) | Ordered list of per-turn API request payloads |

Each object inside `payloads` is sent directly to the server without modification. The loader does not inspect or validate payload contents.

---

## Basic Usage

After running any AIPerf benchmark, an `inputs.json` file is generated in the artifact directory. Replay it:

```bash
aiperf profile \
    --input-file artifacts/my-benchmark/inputs.json \
    --model Qwen/Qwen3-0.6B \
    --custom-dataset-type inputs_json \
    --streaming \
    --url localhost:8000 \
    --concurrency 4
```

Raw payloads work with any endpoint type. The default `chat` endpoint provides structured response parsing (token counts, finish reasons). Use `--endpoint-type raw` only for non-standard APIs where no built-in endpoint matches.

Auto-detection recognizes `inputs.json` files by parsing the full file and matching on the top-level `data` array containing objects with `payloads` keys, so `--custom-dataset-type inputs_json` is optional. Specify it explicitly for reliability -- it skips the auto-detection probe and avoids ambiguity if the file is later edited.

---

## Configuration

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--input-file` | Yes | -- | Path to the inputs JSON file |
| `--model` | Yes | -- | Model name (e.g., `Qwen/Qwen3-0.6B`) |
| `--endpoint-type` | No | `chat` | Any endpoint type works; `raw` available for non-standard APIs |
| `--custom-dataset-type` | No | Auto-detected | Set to `inputs_json` to force this loader |
| `--dataset-sampling-strategy` | No | `shuffle` | `sequential`, `shuffle`, or `random` |
| `--concurrency` | No | -- | Number of concurrent users |
| `--streaming` | No | `false` | Enable streaming responses |

---

## Cross-Server Comparison

Run the same payloads against two different servers to compare performance:

```bash
# Run benchmark against server A
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url server-a:8000 \
    --concurrency 4

# Replay the exact same payloads against server B
aiperf profile \
    --input-file artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency4/inputs.json \
    --model Qwen/Qwen3-0.6B \
    --custom-dataset-type inputs_json \
    --url server-b:8000 \
    --concurrency 4
```

---

## Context Mode

Inputs JSON conversations use `message_array_with_responses` [context mode](../reference/conversation-context-mode.md) by default. Each turn is sent exactly as written -- AIPerf does not accumulate prior turns or inject server responses into subsequent requests.

This is the correct behavior because each payload already contains the complete message history for that point in the conversation.

---

## Comparison with Raw Payload

Both `inputs_json` and `raw_payload` send payloads verbatim, but they differ in structure:

| | `raw_payload` | `inputs_json` |
|--|---------------|---------------|
| Input format | JSONL file or directory of JSONL files | Single JSON file |
| Multi-turn | File mode: no. Directory mode: yes | Yes |
| Session IDs | Auto-generated | Preserved from file |
| Auto-detection | `messages` key in first line | `data` + `payloads` keys |

Choose `inputs_json` when you have a structured file with named sessions (especially from a prior AIPerf run). Choose `raw_payload` when you have flat JSONL logs or a directory of captured conversations.

---

## Tips

- **Prefer `--custom-dataset-type inputs_json`** when replaying AIPerf-generated files. Auto-detection works (the loader parses the full file and matches on `data` + `payloads` keys), but specifying the type explicitly skips the probe and avoids ambiguity if the file is later edited.
- **Payloads are sent verbatim**: The loader does not add, remove, or modify any fields.
- **Turns within a session run sequentially**: Turn 0, then turn 1, etc. Different sessions run concurrently up to `--concurrency`.
- **Check the artifact directory**: After any AIPerf run, look for `inputs.json` -- this is the file you can feed back for replay.
