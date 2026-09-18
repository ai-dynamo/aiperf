---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Replay H CUA Perf Computer-Use Agent Traces
---

# Replay H CUA Perf Computer-Use Agent Traces

AIPerf can replay [H CUA Perf](https://huggingface.co/datasets/Hcompany/h_cua_perf), a dataset of real computer-use agent traces collected by [H Company](https://www.hcompany.ai) while evaluating its agent on [CUA-Gym](https://github.com/xlang-ai/CUA-Gym) desktop tasks. Every record is one chat-completion request the agent actually made, with the agent's recorded think time before it, and the corpus is licensed CC BY 4.0.

Replaying it reproduces the request pattern of a tool-calling, screenshot-driven agent: prompts that grow with every step, a bounded number of images per request, tool definitions on every call, and a shared prefix between consecutive requests that breaks where the agent's context cleaning replaced a screenshot. That makes it a good fit for measuring prefix caching, multimodal prefill, and tool-call parsing under a workload shaped by a real agent rather than a generator.

---

## What Is in the Dataset

477 trajectories and 18,224 requests, one per agent step, 38.2 steps per trajectory on average and up to 201. Each trajectory is a different task, and the corpus is balanced between successes and failures and across five step-count buckets. The [dataset card](https://huggingface.co/datasets/Hcompany/h_cua_perf) covers the collection, the curation and the record schema; what follows is what shapes a benchmark run.

- **Closed-loop by default.** A record carries the `delay` since the previous response but no absolute timestamp, so the corpus replays at a chosen concurrency rather than on a fixed schedule. A trajectory's records are contiguous, and `session_id` is the task id.
- **One screenshot per request.** The published build keeps the latest screenshot only; earlier ones are the text placeholder `[Image omitted by context cleaning]` that the agent's own context cleaning left. The `n_screenshots` filter restores a wider window.
- **Tool calls on nearly every request.** All but 25 requests carry 10 tool definitions and `tool_choice: auto`, so the server needs a tool parser configured.
- **Large, growing records.** The prompt holds the full text history of the trajectory, so its length grows roughly linearly with the step index while the image count stays at one. The average record is 0.5 MB and the largest 2.6 MB.

---

## Start a vLLM Server

The traces expect a vision-capable chat model that accepts OpenAI tool definitions. Launch vLLM with tool calling enabled and a large context window, because prompts grow with the step index:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 -e HF_TOKEN vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 131072 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

`--enable-auto-tool-choice` and `--tool-call-parser` are required: every request sends `tools` and `tool_choice: auto`, and vLLM rejects those with HTTP 400 unless a tool parser is configured.

Verify the server is ready:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-VL-7B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Profile with the Public Dataset

AIPerf downloads the manifest `h_cua.meta.json` and the trace `h_cua.jsonl.zst` (2.5 GB compressed, 9.3 GB decompressed) into the HuggingFace Hub cache (`~/.cache/huggingface/hub`, or `HF_HOME`) on first use and reads them from there afterwards. The dataset is public, so no authentication is required; if your account cannot see it, run `uv run hf auth login` or set `HF_TOKEN`. The download plus parsing takes longer than the default 300 s dataset-configuration timeout, so raise both timeouts (`AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` must be greater than or equal to `AIPERF_DATASET_CONFIGURATION_TIMEOUT`).

Start with a subset of trajectories. `--num-dataset-entries N` keeps the first N trajectories and stops reading the file there, so the remaining gigabytes are never decompressed or parsed:

```bash
AIPERF_DATASET_CONFIGURATION_TIMEOUT=3600 \
AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=3600 \
aiperf profile \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --public-dataset h_cua_perf \
    --num-dataset-entries 20 \
    --num-conversations 20 \
    --concurrency 4 \
    --inter-turn-delay-cap-seconds 5
```

Each trajectory is replayed as one multi-turn conversation: AIPerf sends a step's request, waits for the live response, sleeps the recorded think time, then sends the next step's request with its recorded `messages`. `--concurrency 4` keeps four trajectories in flight. `--num-conversations` sets how many trajectories to play; `--num-dataset-entries` sets how many to load, and is the one that bounds the parse of the source file.

Recorded delays are respected by default: the agent paused for as long as its tool calls took to run, which can be tens of seconds, and AIPerf sleeps for those gaps faithfully. `--inter-turn-delay-cap-seconds` clamps them; capping makes a run finish in reasonable time, not capping is the more honest replay. To send each trajectory's requests back to back, as soon as the previous response arrives, pass `--inter-turn-delay-cap-seconds 0`. `--ignore-trace-delays` and `--use-think-time-only` act on the Weka loaders only and have no effect on this dataset.

Without `--num-dataset-entries` or a trajectory filter, the full corpus is loaded and held as parsed records while conversations are built. Parsed JSON takes more memory than its on-disk size, so plan for RAM well above the 9.3 GB decompressed size; this is an untested estimate, not a measurement. AIPerf also writes `inputs.json`, the payload of every loaded request, into the artifact directory, as it does for other public datasets. Here that file is roughly the size of the loaded records (about 0.5 MB per request), which is one more reason to start from a subset. The 2.5 GB download stays in the Hub cache and is read once per run to hash it, which takes seconds.

---

## Selecting and Reshaping Trajectories

The dataset ships `trace_processor.py`, a script that derives replay variants from the source: how many of the latest screenshots each request keeps, and which trajectories to include at what length. The public-dataset loader implements the same options in place through `--dataset-filter`, so no derived file is needed:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --public-dataset h_cua_perf \
    --num-dataset-entries 100 \
    --dataset-filter n_screenshots=3 \
    --dataset-filter avg_trace_length=20 \
    --num-conversations 100 \
    --concurrency 4
```

| Filter | Default | Effect |
| --- | --- | --- |
| `n_screenshots` | unset | Each request keeps its own screenshot plus the preceding ones of its trajectory, restored from the records that carry them; older ones stay as the placeholder text. Unset leaves the published single screenshot per request. |
| `min_trace_length` | `1` | Drop trajectories with fewer requests; also the floor of every truncation. |
| `max_trace_length` | unset | Truncate trajectories to their first N requests. |
| `avg_trace_length` | unset | Target mean requests per trajectory, reached by truncating every trajectory by the same factor (a bisection on the factor, floored at `min_trace_length`). |

The selection is planned from the manifest before any record is parsed, in the script's order: length filter, truncation, the `--num-dataset-entries` cut, then the average fit. Only the selected trajectories are read from the compressed file, and reading stops after the last one. The names, defaults, and rounding are the script's, so a filter set and the script run with the same options on the same trajectories produce the same records; the loader logs the resulting trajectory and request counts.

The script's `--num-traces` and `--seed` random sample has no in-place equivalent. `--num-dataset-entries` keeps the first N eligible trajectories in file order, where file order is by task id and therefore unrelated to length or outcome; to replay a seeded random sample, derive a file with the script instead.

A wider screenshot window multiplies the bytes held per request, in memory and in `inputs.json`, and needs a matching server limit, for example `--limit-mm-per-prompt '{"image":3}'` on vLLM for `n_screenshots=3`. Pair it with `--num-dataset-entries` or `max_trace_length`.

---

## What Is Reproduced, and What Is Not

Every request goes out as recorded: the `messages` verbatim, the tool definitions, the recorded completion length as `max_completion_tokens` (or `max_tokens` with `--use-legacy-max-tokens`), and the recorded think time before it. What a replay cannot reproduce:

- **The model's own responses.** Each request replays the recorded history verbatim, so what the server generates at step N is not fed into step N+1. The trajectory follows the recorded agent, not the model under test.
- **Tool execution.** Tools are not run; the recorded observations are replayed.
- **Earlier screenshots.** Only the latest screenshot per request is an image, unless `n_screenshots` widens the window.

---

## Replaying From a File

The same variants can be materialized as files with the dataset's script, for example to share a fixed subset or to replay offline:

```bash
uv run hf download Hcompany/h_cua_perf --repo-type dataset --local-dir h_cua_perf
pip install tyro pydantic zstandard

# Every request keeps its 3 latest screenshots
python h_cua_perf/trace_processor.py \
    --input-path h_cua_perf/h_cua.jsonl.zst \
    --output-path h_cua-3shots.jsonl \
    --n-screenshots 3

# 100 trajectories truncated to average 20 requests, single screenshot
python h_cua_perf/trace_processor.py \
    --input-path h_cua_perf/h_cua.jsonl.zst \
    --output-path h_cua-small.jsonl \
    --n-screenshots 1 --num-traces 100 --avg-trace-length 20 --seed 0
```

Derived files are plain Mooncake trace JSONL and replay through the file-based loader:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --custom-dataset-type mooncake_trace \
    --input-file h_cua-3shots.jsonl \
    --num-conversations 20 \
    --concurrency 4
```

`--public-dataset h_cua_perf` with a set of filters and the script run with the same options on the same trajectories, followed by `--input-file`, produce identical conversations: the public loader only adds the download and the in-place selection, and delegates parsing and turn construction to the Mooncake trace loader. One difference remains. The file-based loader reads plain JSONL, so the source build itself has to be decompressed first (`zstd -d h_cua.jsonl.zst`), and neither `--num-dataset-entries` nor the trajectory filters apply to it; make a smaller file with the script instead.

---

## Record Format

A record holds `session_id`, `delay`, `messages`, `tools`, `output_length` and `extra`. The [dataset card](https://huggingface.co/datasets/Hcompany/h_cua_perf#record-format) has a worked example, and [Trace Replay with Mooncake Traces](../benchmark-modes/trace-replay.md) is the Mooncake field reference.

Two fields are handled outside the `messages` array: `tools` is sent alongside them, and `extra` is shallow-merged into the request body. Everything else is passed through untouched, including the `uuid` on each image part (xxh3-64 of the data URL), which lets a replay send uuid-only references to a server that already holds the image.

---

## Related Tutorials

- [Trace Replay with Mooncake Traces](../benchmark-modes/trace-replay.md)
- [Multi-Turn Conversations](multi-turn.md)
- [Replay Weka Agentic Coding Traces](weka-trace.md)
- [Profile Vision Language Models](vision.md)
