<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Tool / function calls (`--tool-call-*`)

Make a seeded fraction of **chat** requests answer with a function tool call instead of a
plain assistant turn, so the runner's tool-call parsing and streamed-argument merge are
exercised. Only affects the OpenAI-compatible chat endpoint; every other front door is
unchanged. Source: `rust/mock-server/src/config.rs`, `ToolCallSpec` / `tool_call_frames` /
`build_chat_response` in `rust/mock-server/src/handlers.rs`.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--tool-call-rate <0..1>` | 0.0 | Seeded probability a chat request emits a function tool call (per-request draw from the `mock.tool_calls` stream). `0.0` disables it (payload byte-unchanged) |
| `--tool-call-name <str>` | `get_weather` | Function name for emitted calls |
| `--tool-call-arguments <json-string>` | `{"location":"NYC"}` | Argument string sent verbatim as `function.arguments` (OpenAI encodes arguments as a JSON *string*, not an object) |

When a tool call fires: the finish reason becomes `tool_calls`, and the emitted `usage`
carries `toolUsePromptTokenCount` (the mock-tokenized length of `name + arguments`). The
generated content stays alongside the call (the token/latency model is unchanged).

## Non-streaming wire shape

`choices[0].message.tool_calls` is a one-element array:

```json
{"id": "call_<uuid>", "type": "function",
 "function": {"name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}}
```

`choices[0].finish_reason == "tool_calls"`.

## Streaming wire shape (two `delta.tool_calls` frames)

To exercise the runner's argument-concatenation merge, the arguments are split across two
`delta.tool_calls` frames (merged by streamed `index`):

- **Frame 1 (open):** `delta.tool_calls[0]` carries `index:0`, `id`, `type:"function"`,
  `function.name`, and the first half of `arguments`. `finish_reason: null`. If no content or
  reasoning token preceded it, this frame also stamps `role: "assistant"`.
- **Frame 2 (close):** `delta.tool_calls[0]` carries `index:0`, the second half of `arguments`
  (no `id`/`type`/`name`), and `finish_reason: "tool_calls"`.

## e2e recipes (`test_tool_calls.rs`)

Mock config: `--tool-call-rate 1.0 --tool-call-name get_weather --tool-call-arguments
'{"location":"NYC"}' --fast --no-tokenizer --random-seed 7`.

Streaming:

```bash
aiperf profile --model gpt-4 --url http://127.0.0.1:8000 --endpoint-type chat --streaming \
  --use-server-token-count \
  --input-file prompts.jsonl --custom-dataset-type single_turn \
  --request-count 6 --concurrency 2 --workers-max 1 \
  --random-seed 7 --export-level raw --ui simple
```

Raw-record assertions: reconstructed streamed call has `function.name == "get_weather"`,
merged `function.arguments == {"location":"NYC"}`; a frame carries `finish_reason == "tool_calls"`;
streamed `content` non-empty; the terminal usage frame has `toolUsePromptTokenCount > 0`.

Non-streaming (drop `--streaming` and `--use-server-token-count`): asserts
`choices[0].finish_reason == "tool_calls"`, `tool_calls[0].type == "function"`,
`tool_calls[0].id` starts with `"call_"`, `function.name == "get_weather"`,
`function.arguments == {"location":"NYC"}`, `usage.toolUsePromptTokenCount > 0`.

Summary: `usage_tool_use_prompt_tokens.avg > 0`.
