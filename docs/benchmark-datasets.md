---
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Benchmark Datasets
---

This document describes datasets that AIPerf can use to generate stimulus. Additional support is under development, so check back often.

## Dataset Options

<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="width:15%; text-align: left;">Dataset</th>
      <th style="width:10%; text-align: center;">Support</th>
      <th style="width:65%; text-align: left;">Data Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Synthetic Text</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated text prompts pulled from Shakespeare</td>
    </tr>
    <tr>
      <td><strong>Synthetic Audio</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated audio samples</td>
    </tr>
    <tr>
      <td><strong>Synthetic Images</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated image samples</td>
    </tr>
    <tr>
      <td><strong>Custom Data</strong></td>
      <td style="text-align: center;">✅</td>
  <td>--input-file your_file.jsonl --custom-dataset-type single_turn</td>
    </tr>
    <tr>
    <td><strong>Mooncake</strong></td>
    <td style="text-align: center;">✅</td>
    <td>Mooncake trace file <a href="benchmark-modes/trace-replay.md"><code>--input-file your_trace_file.jsonl --custom-dataset-type mooncake_trace</code></a></td>
    </tr>
    <tr>
    <td><strong>Baseten Trace</strong></td>
    <td style="text-align: center;">✅</td>
    <td>Baseten completion trace parquet <a href="tutorials/baseten-trace.md"><code>--input-file your_trace.parquet --custom-dataset-type baseten_trace</code></a></td>
    </tr>
    <tr>
      <td><strong>ShareGPT</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Conversations from <a href="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"><code>--public-dataset sharegpt</code></a></td>
    </tr>
    <tr>
      <td><strong>Hugging Face trajectories</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Recorded trajectories from <a href="#hugging-face-mooncake-trajectory-replay"><code>--public-dataset mooncake_hf</code></a></td>
    </tr>
    <tr>
      <td><strong>Exgentic</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Recorded agent sessions from <a href="https://huggingface.co/datasets/Exgentic/agent-llm-traces"><code>--public-dataset exgentic</code></a></td>
    </tr>
    <tr>
      <td><strong>Exgentic v2</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Expanded recorded agent sessions from <a href="https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2"><code>--public-dataset exgentic_v2</code></a></td>
    </tr>
    <tr>
      <td><strong>Agentic Code</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetic multi-turn coding-agent traces with shared prompt layers, repository context, and cache-aware turn growth. Generated via <a href="tutorials/agentic-code-generator.md"><code>aiperf synthesize agentic-code</code></a> and replayed as a Mooncake trace.</td>
    </tr>
    <tr>
      <td><strong>H CUA Perf</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Real computer-use agent traces (tool calling, one screenshot per request, recorded think time) from <a href="https://huggingface.co/datasets/Hcompany/h_cua_perf">H Company</a>, replayed as a Mooncake trace <a href="tutorials/h-cua-perf.md"><code>--public-dataset h_cua_perf</code></a></td>
    </tr>
    <tr>
      <td><strong>TraceLab</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Real agentic coding sessions from the <a href="https://github.com/uw-syfi/TraceLab">TraceLab</a> corpus, replayed with synthesized KV-cache block IDs and recovered subagent nesting <a href="tutorials/tracelab-trace.md"><code>--input-file syfi_coding_trace.jsonl.gz --custom-dataset-type tracelab</code></a></td>
    </tr>
  </tbody>
</table>

## Exgentic Agent Trace Replay

The Exgentic loaders stream recorded agent sessions directly from Hugging Face. `exgentic` is pinned to v1 revision `70036b93a04e61b0ea2706a68b962f4f26774587`; `exgentic_v2` is independently pinned to v2 revision `4b8ad4ab198438e5a170f9171c19c6a2cf7c1814`. Each replays successful, positive-token chat call snapshots. Recorded messages, system instructions, tool definitions, output-token limits, request controls, and call start times are preserved. Tools are not executed, and live responses are not added to later requests. Every request carries the source session as `x-dynamo-session-id` for Dynamo agentic tracing while AIPerf retains its own request correlation ID.

Provide a finite materialization bound through `--num-conversations`, `--num-dataset-entries`, or `--request-count`. `--benchmark-duration` limits request issuance, not dataset setup.

Select a source harness and source model independently from the target model served by the endpoint:

```bash
aiperf profile \
  --model TARGET_MODEL \
  --url http://localhost:8000/v1/chat/completions \
  --endpoint-type chat \
  --public-dataset exgentic_v2 \
  --dataset-filter benchmark=swebench \
  --dataset-filter harness=tool_calling \
  --dataset-filter source_model=Kimi-K2.5 \
  --num-conversations 1 \
  --fixed-schedule
```

`source_model` selects the model that produced the trace; `--model` selects the target model receiving the replay. `benchmark` selects an Exgentic v2 workload. Invalid filters report the available harness/model combinations. The v1 dataset contains 22 combinations across five harnesses and six canonical source models.

Fixed-schedule mode emits each recorded call as an independently scheduled one-turn request using its start offset from the source session. Calls that overlapped in the trace therefore overlap during replay. Selected source sessions start together at offset zero. Without `--fixed-schedule`, each source session remains a closed-loop multi-turn conversation: AIPerf waits for one live response before applying the recorded residual delay and sending the next request.

Size the target context window for the selected trace plus the target model's chat-template overhead. Recorded contexts reach about 178K tokens, and some Gemini tool-calling sessions exceed 64K before target formatting. `tool_calling_with_shortlisting` alternates a selector request containing the full tool catalog with executor requests containing a changing subset of schemas. Low executor prefix-cache reuse is expected for that harness and is not a loader error.

## Hugging Face Mooncake Trajectory Replay

Use `--public-dataset mooncake_hf` to replay complete recorded trajectories from a Hugging Face repository selected with `--hf-mooncake-dataset`. This flag automatically selects `mooncake_hf`.
The loader extends the Hugging Face reader and delegates to Mooncake's recorded request snapshots (`message_array_with_responses`). This mode expects **one complete trajectory per row**.

For example, map [sammshen/lmcache-agentic-traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces) using explicit source columns:

```bash
aiperf profile \
  --model MiniMaxAI/MiniMax-M2.5 \
  --url http://localhost:8000/v1/chat/completions \
  --endpoint-type chat \
  --hf-mooncake-dataset sammshen/lmcache-agentic-traces \
  --dataset-filter model=minimax-m2.5 \
  --mooncake-trace-to-hf-mapping messages=input \
  --concurrency 64 \
  --num-sessions 100 \
  --streaming
```

Each `--mooncake-trace-to-hf-mapping TARGET=SOURCE` maps a Mooncake field to a dotted source path.
Supported targets are `session_id`, `messages`, `tools`, `output_length`, `extra`, and `extra.PARAMETER` for a top-level request parameter.
Unmapped fields use their canonical Mooncake column names.

Each HF row remains one session. The loader creates a request before **every assistant message**, including tool-call-only messages.
For example, `[system, user, assistant, tool, assistant, tool, assistant, tool]` produces:

1. `[system, user]`
2. `[system, user, assistant, tool]`
3. `[system, user, assistant, tool, assistant, tool]`

Use concurrency or request-rate scheduling. Fixed scheduling is rejected because these trajectories contain no per-call timestamps.
Requests within each session run sequentially, while separate sessions can run concurrently. `--num-sessions` counts trajectories; `--request-count` counts individual generation steps and can stop partway through a trajectory. Use the selected trajectory count to replay each once with sequential sampling.
