---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Replay H CUA Perf Computer-Use Agent Traces
---

# Replay H CUA Perf Computer-Use Agent Traces

[H CUA Perf](https://huggingface.co/datasets/Hcompany/h_cua_perf) is 477 sessions (18,224 requests) recorded from [H Company](https://www.hcompany.ai)'s computer-use agent on [CUA-Gym](https://github.com/xlang-ai/CUA-Gym) desktop tasks. License: CC BY 4.0.

Each record is one real chat-completion request:

- the full text history of the session so far
- 10 tool definitions
- the latest screenshot
- the recorded completion length
- the time the agent waited before sending it, tool execution included

Prompts grow with every turn and a sliding window bounds the screenshots per request, so the replay exercises prefix caching, multimodal prefill and tool-call parsing the way a real agent does. The [dataset card](https://huggingface.co/datasets/Hcompany/h_cua_perf) describes the record schema.

## Server

The server needs a vision model, a tool parser, a context window as large as the model allows and an image limit at least as large as the screenshot window.

```bash
docker run --gpus all -p 8000:8000 -e HF_TOKEN vllm/vllm-openai:latest \
  Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 128000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --limit-mm-per-prompt '{"image":5}' \
  --enable-prompt-tokens-details
```

`--enable-prompt-tokens-details` is what lets AIPerf report prompt-cache hits.

## Replay

```bash
AIPERF_DATASET_CONFIGURATION_TIMEOUT=3600 \
AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=3600 \
aiperf profile \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --use-server-token-count \
    --public-dataset h_cua_perf \
    --num-dataset-entries 20 \
    --dataset-filter n_screenshots=3 \
    --dataset-filter max_trace_length=40 \
    --num-conversations 20 \
    --concurrency 4
```

| Flag | Effect |
| --- | --- |
| `--concurrency N` | Sessions in flight at once. The turns of a session are sequential, so this is the number of agents working at the same time. Always set it, the default is 1. |
| `--inter-turn-delay-cap-seconds` | Caps the recorded wait between two turns of a session. Unset, the waits are replayed as recorded; `0` sends each session's requests back to back and the run becomes a plain concurrency test. |
| `--num-conversations N` | Stops after N whole sessions. |
| `--request-count N` | Stops after N requests, cutting the last sessions short. Without either cap, AIPerf stops after 10 requests. |
| `--use-server-token-count` | Takes token counts from the server's usage. AIPerf's own count tokenizes text only and ignores images, so without it the input sequence length misses every screenshot. |

Some requests are larger than the 128k-token context of the example server; `max_trace_length=40` and the small screenshot window keep the example's requests under it.

Each session is one multi-turn conversation: a request is sent, the response awaited, the recorded wait slept, then the next recorded request is sent. There are no timestamps, so `--fixed-schedule` does not apply.

The screenshot window and the session selection are `--dataset-filter` options:

| Filter | Effect |
| --- | --- |
| `n_screenshots` | Sliding window of the N latest screenshots per request; the published records carry one. |
| `min_trace_length` | Drop shorter sessions; truncations never go below it (default 1). |
| `max_trace_length` | Keep each session's first N turns. |
| `avg_trace_length` | Scale every session's length by the same factor until the mean reaches the target. |

## Related Tutorials

- [Trace Replay with Mooncake Traces](../benchmark-modes/trace-replay.md)
- [Multi-Turn Conversations](multi-turn.md)
