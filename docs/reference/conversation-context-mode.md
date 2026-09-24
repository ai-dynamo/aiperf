---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Conversation Context Mode
---

# Conversation Context Mode

Conversation context mode controls how prior turns are accumulated when building multi-turn chat requests. Different dataset formats imply different accumulation strategies, and AIPerf automatically selects the right one based on your data.

Two dimensions determine the mode:

- **Turn format**: `DELTAS` (incremental per-turn content) vs `MESSAGE_ARRAY` (each turn carries its complete message list)
- **Response inclusion**: `WITH_RESPONSES` (assistant replies supplied by the dataset) vs `WITHOUT_RESPONSES` (replies generated during the run are captured and retained)

## Modes

### `deltas_without_responses`

Standard multi-turn chat. Each dataset turn supplies new input messages. AIPerf accumulates turns and threads live inference responses into the history. The initial history may include recorded assistant messages from before replay begins; those messages are preserved too.

**Dataset:**
```
Turn 1: {"role": "user", "content": "What is ML?"}
Turn 2: {"role": "user", "content": "Give an example"}
Turn 3: {"role": "user", "content": "How does it differ from traditional programming?"}
```

**Replay:**
```
Request 1: [User "What is ML?"]
  → Server responds with A1

Request 2: [User "What is ML?", Assistant A1, User "Give an example"]
  → Server responds with A2

Request 3: [User "What is ML?", Assistant A1, User "Give an example", Assistant A2, User "How does it differ..."]
  → Server responds with A3
```

Default for:
- Synthetic datasets
- Multi-turn JSONL
- ShareGPT
- Mooncake traces with `text_input` or synthesized `input_length` (including `hash_ids`)

Opt-in for:
- Mooncake traces with `messages` arrays marked `"assistant_responses": "live"` on every row in the session (see [Trace Replay](../benchmark-modes/trace-replay.md#live-responses-for-structured-messages)). The first entry carries the initial history, including any earlier recorded assistant messages; each later entry carries only the new messages for that turn. AIPerf inserts live assistant responses (text and tool calls) between entries. Use `--custom-dataset-type mooncake_trace` to select this format explicitly.

### `deltas_with_responses`

Delta-compressed prompts. Each dataset turn only contains the *new* messages since the previous turn. AIPerf accumulates these deltas to reconstruct the full conversation. The live inference response is only used for measurement and discarded -- the pre-canned assistant responses in the dataset are used instead.

**Dataset (each turn is a delta):**
```
Turn 1: [{"role": "user", "content": "What is ML?"}]
Turn 2: [{"role": "assistant", "content": "ML is..."}, {"role": "user", "content": "Give an example"}]
Turn 3: [{"role": "assistant", "content": "Sure..."}, {"role": "user", "content": "How does it differ..."}]
```

**Replay (deltas accumulated):**
```
Request 1: [User "What is ML?"]
  → Live response discarded

Request 2: [User "What is ML?"] + [Assistant "ML is...", User "Give an example"]
  → Live response discarded

Request 3: [User "What is ML?"] + [Assistant "ML is...", User "Give an example"] + [Assistant "Sure...", User "How does it differ..."]
  → Live response discarded
```

Default for:
- N/A (no built-in loader defaults to this mode yet)

### `message_array_with_responses`

Self-contained prompts. Each turn already contains its full context (including assistant responses). No session accumulation.

**Dataset:**
```
Turn 1: [{"role": "user", "content": "What is ML?"}]
Turn 2: [{"role": "user", "content": "What is ML?"}, {"role": "assistant", "content": "ML is..."}, {"role": "user", "content": "Give an example"}]
Turn 3: [{"role": "user", "content": "What is ML?"}, {"role": "assistant", "content": "ML is..."}, {"role": "user", "content": "Give an example"}, {"role": "assistant", "content": "Sure..."}, {"role": "user", "content": "How does it differ..."}]
```

**Replay:**
```
Request 1: sends Turn 1 as-is
Request 2: sends Turn 2 as-is
Request 3: sends Turn 3 as-is
```

Each turn is sent exactly as it appears in the dataset.

Default for:

- Mooncake traces with pre-built `messages` arrays (the default `"assistant_responses": "recorded"`)
- Bailian traces: a follow-up row's `input_length` and `hash_ids` already cover the history the service retained plus the new text, so each row is sent as its own generated prompt


### `message_array_without_responses`

Reserved for future use. Each turn would carry a complete user-only message array, requiring live response merging between turns. Not yet implemented.

## How It Works

Context mode is resolved through a priority chain:

1. **Per-conversation override** -- A conversation in the dataset can specify its own `context_mode`
2. **Loader default** -- The dataset loader can declare a default based on dataset format semantics
3. **Global fallback** -- `deltas_without_responses`

This means most users never need to think about context mode. The loader picks the right default, and individual conversations can override it when needed.
