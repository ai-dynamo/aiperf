---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Replay H CUA Perf Computer-Use Agent Traces
---

# Replay H CUA Perf Computer-Use Agent Traces

[H CUA Perf](https://huggingface.co/datasets/Hcompany/h_cua_perf) is 477 trajectories (18,224 requests) of a computer-use agent, recorded by [H Company](https://www.hcompany.ai) on [CUA-Gym](https://github.com/xlang-ai/CUA-Gym) desktop tasks and licensed CC BY 4.0. Each record is a real chat-completion request: the full text history of the trajectory, 10 tool definitions with `tool_choice: auto`, the latest screenshot as a PNG data URL, the recorded completion length, and the agent's think time since the previous response. Prompts grow with every step while the number of screenshots per request is bounded by a sliding window you choose, so the workload exercises prefix caching, multimodal prefill and tool-call parsing with a real agent's request pattern. The [dataset card](https://huggingface.co/datasets/Hcompany/h_cua_perf) describes the collection and the record schema.

## Server

The server needs a vision model, a large context window and a tool parser; without the parser vLLM rejects `tool_choice: auto` with HTTP 400.

```bash
docker run --gpus all -p 8000:8000 -e HF_TOKEN vllm/vllm-openai:latest \
  Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 128000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --limit-mm-per-prompt '{"image":5}' \
  --enable-prompt-tokens-details
```

`--limit-mm-per-prompt` must allow as many images as the screenshot window below, and `--enable-prompt-tokens-details` is what lets AIPerf report prompt-cache hits. Long trajectories contain a few memory-compressor requests whose prompt plus recorded completion exceeds 128k tokens: the server answers those with HTTP 400, AIPerf counts them as failed requests and the run completes. That is expected, not a misconfiguration.

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
    --dataset-filter avg_trace_length=20 \
    --num-conversations 20 \
    --concurrency 4
```

`--public-dataset h_cua_perf` downloads `h_cua.jsonl.zst` (2.5 GB) and its manifest into the HuggingFace Hub cache on first use. The two timeouts cover that first download and are unnecessary afterwards. An authenticated endpoint takes `--api-key`. `--use-server-token-count` matters: AIPerf's built-in token count tokenizes text only and ignores images, so without it the input sequence length misses every screenshot.

Each trajectory is replayed as one multi-turn conversation: a step's request is sent, the response awaited, the recorded think time slept, then the next step's recorded `messages` are sent. `--concurrency` is the number of trajectories in flight and `--num-conversations` how many are played. Think times are respected by default and can reach tens of seconds; `--inter-turn-delay-cap-seconds` caps them, and a cap of `0` sends each trajectory's requests back to back (`--ignore-trace-delays` applies to the Weka loaders only). There are no absolute timestamps, so `--fixed-schedule` does not apply.

`tools` is sent with the messages, `extra` is merged into the request body, and everything inside `messages` goes out untouched, including the `uuid` on image parts. Model responses are not fed into later requests and tools are not executed: the trajectory follows the recorded agent, not the model under test. Completions come from the model under test, capped at the recorded `output_length`, so their length rarely matches the recording and decode-side metrics say little about the agent's workload; prefill and prompt-cache metrics are what this dataset exercises.

## Screenshot Window

The published build keeps one screenshot per request, the latest; every earlier one was replaced in place by the text `[Image omitted by context cleaning]`, exactly as the agent's own context cleaning did when it ran. `--dataset-filter n_screenshots=N` restores a sliding window: each request goes out with its own screenshot plus the N-1 preceding ones of its trajectory, taken from the records that carry them, and anything older stays as the placeholder. The window is the main knob of this benchmark, because it sets two things at once:

- **Image load per request.** Screenshots dominate request bytes and multimodal prefill; N images per request is N times that cost, on every step of every trajectory.
- **Where the shared prefix breaks.** Consecutive requests of a trajectory are identical up to the first screenshot the window has since dropped. With N=1 that is the previous step's observation, so nearly the whole prompt is a cache hit; with N=3 the placeholder lands three steps back, so the reusable prefix ends earlier and the server re-prefills the last three observations. Widening the window therefore trades prefix-cache hits for image tokens, which is the shape a real agent with a wider context window imposes.

Unset, the loader replays the published single screenshot. A wider window multiplies `inputs.json` and the size of every request on the wire, not the loader's memory: restored screenshots are shared by reference, and RAM is driven by how many trajectories are loaded and how long they are. Read the effect in the input sequence length, `Overall Usage Prompt Cache Read %` and TTFT.

## Selecting Trajectories

Selection is planned from the manifest before any record is parsed, and reading stops after the last selected trajectory. `--num-dataset-entries N` keeps the first N trajectories in file order. Without it the full corpus is loaded, about twice its 9.3 GB decompressed size once parsed; the loader warns before reading when the selection will not fit in the available memory.

`--dataset-filter` reshapes the selection with the options of the dataset's `trace_processor.py`, applied in place, so a filter set yields the same records as the script run with the same options on the same trajectories. The script's seeded random sample (`--num-traces`) has no in-place equivalent.

| Filter | Effect |
| --- | --- |
| `n_screenshots` | Sliding window of the N latest screenshots per request (see above). |
| `min_trace_length` | Drop shorter trajectories; also the floor of every truncation (default 1). |
| `max_trace_length` | Keep each trajectory's first N requests. |
| `avg_trace_length` | Truncate every trajectory by one common factor until the mean length reaches the target. |

## Related Tutorials

- [Trace Replay with Mooncake Traces](../benchmark-modes/trace-replay.md)
- [Multi-Turn Conversations](multi-turn.md)
