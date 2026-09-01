---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile Native SGLang Generate Endpoints
---

# Profile Native SGLang Generate Endpoints

Use the `sglang_generate` endpoint type to benchmark a native SGLang-compatible streaming `/generate` API with token IDs instead of OpenAI text or message payloads. This path is useful for measuring a frontend or router without server-side prompt tokenization.

## Request and Response Shape

AIPerf sends cumulative conversation context as `input_ids`:

```json
{
  "rid": "request-id",
  "input_ids": [1, 2, 3],
  "sampling_params": {
    "max_new_tokens": 128,
    "ignore_eos": true
  },
  "priority": 192,
  "stream": true
}
```

The endpoint must stream SGLang-style SSE events containing incremental `output_ids` and `meta_info`, followed by `[DONE]`:

```text
data: {"output_ids":[42],"meta_info":{"prompt_tokens":3,"completion_tokens":1,"finish_reason":null}}

data: [DONE]
```

AIPerf stores streamed output IDs as the assistant turn and appends them to the next request in the same session. Input text is tokenized once while the dataset is built, not on the request-dispatch hot path. Hash-ID trace prompts preserve the exact token sequence used before prompt decoding.

## Profile a Mooncake Trace

```bash
aiperf profile \
  --model Qwen/Qwen3-30B-A3B \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --endpoint-type sglang_generate \
  --streaming \
  --url http://localhost:8000 \
  --input-file trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --no-fixed-schedule \
  --concurrency 512 \
  --session-header X-Dynamo-Session-ID \
  --use-server-token-count
```

`--session-header X-Dynamo-Session-ID` sends each conversation's stable correlation ID in the header Dynamo uses for session affinity. `--no-fixed-schedule` is appropriate when the trace timestamps should not control arrivals and concurrency should instead bound the number of live sessions.

Mooncake `extra.nvext.agent_hints.strict_priority` is mapped to the top-level SGLang `priority` field. Other per-turn extras remain top-level request fields, while a `sampling_params` object is merged into the generated sampling parameters.

## Limitations

- Streaming is required. AIPerf rejects this endpoint type unless `--streaming` is set.
- Each request turn must contain text/hash-ID input or pre-populated token IDs. Raw `messages` and raw `payload` Mooncake modes are not supported.
- Conversation-level system and user-context prompts are not supported because `/generate` receives token IDs directly. Include that content in the dataset turns before benchmarking.
- The endpoint sends one sequence per request; SGLang parallel sampling is outside this endpoint's scope.
