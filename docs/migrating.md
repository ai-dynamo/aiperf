<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Migrating from GenAI-Perf

AIPerf is designed to be a drop-in replacement for [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf) _for currently supported features_. Most options from GenAI-Perf map directly to AIPerf options. Options that don't are noted below.
Some options, primarily for the `analyze` subcommand, are not yet supported; they're planned for future releases.
<br>

See the [GenAI-Perf vs AIPerf CLI Feature Comparison Matrix](genai-perf-feature-comparison.md) for a detailed comparison of the supported CLI options.

## Known CLI Argument Differences

- `--max-threads`: You no longer need to set a max-thread option. Previously, this was a global setting to control GenAI-Perf total thread count.
AIPerf provides more-fine grained control of the number of workers issuing requests to the endpoint by using the `--workers-max` option.
- `--`: The passthrough args flag is no longer required. All options are now natively supported by AIPerf.

To migrate your previous GenAI-Perf commands to AIPerf commands, remove the above options.

<br>

## Input File Format Changes

The format for the `inputs.json` file, which contains the input prompts used in benchmarking, has changed slightly from GenAI-Perf to AIPerf:

- **`payload` → `payloads`**: The singular `payload` field has been renamed to `payloads` (plural).
- **Conversation turns**: Each item in the `payloads` array now represents a turn in a conversation/session, providing better support for multi-turn interactions.
- **`session_id` field**: A new `session_id` field has been added to each entry. This enables correlation between requests and payloads for future analytics and tracking purposes.

These changes allow AIPerf to better handle conversational workloads and provide more detailed traceability for performance analysis.

<br>

## Reasoning Tokens and Differences in Metrics

Modern language models with reasoning capabilities such as openai/gpt-oss-120b, DeepSeek-R1, Qwen3, or similar, generate reasoning tokens before producing their final response. These reasoning tokens are typically returned in the `reasoning_content` field of the API response and represent the model's internal thought process.

### Behavioral Differences

> [!IMPORTANT]
> **GenAI-Perf does not parse or process reasoning tokens**. Content in the `reasoning_content` field is ignored, which means GenAI-Perf waits until the first non-reasoning output token is generated before recording the Time to First Token (TTFT).

**AIPerf** fully supports parsing and processing of reasoning tokens. The TTFT metric captures the time to generate the first token of any type, whether it's a reasoning token or an output token. Additionally, AIPerf introduces a new metric: **Time to First Output Token (TTFO)**, which measures the time to the first non-reasoning output token, equivalent to GenAI-Perf's TTFT.

### Impact on Metrics

When comparing benchmark results between the two tools for reasoning-capable models:

#### Time to First Token (TTFT)

> [!TIP]
> When migrating from GenAI-Perf, use **AIPerf TTFO** to compare against **GenAI-Perf TTFT** for equivalent measurements of reasoning-capable models.

- **AIPerf TTFT** measures time to the first token of any type (including reasoning tokens) from the start of the request, and will be lower than GenAI-Perf TTFT
- **GenAI-Perf TTFT** measures time to the first non-reasoning output token from the start of the request, and will be higher than AIPerf TTFT
- **AIPerf TTFO** is equivalent to GenAI-Perf TTFT

By providing both TTFT and TTFO metrics, AIPerf enables more comprehensive performance analysis of reasoning-capable models by offering complete visibility into the token generation timeline.

#### Output Sequence Length (OSL)

> [!TIP]
> When migrating from GenAI-Perf, use **AIPerf Output Token Count** to compare against **GenAI-Perf OSL** for equivalent measurements of reasoning-capable models.

- **AIPerf OSL** includes both reasoning and output tokens, so it will be higher than GenAI-Perf OSL
- **GenAI-Perf OSL** excludes reasoning tokens from the count, so it will be lower than AIPerf OSL
- **AIPerf Output Token Count** is equivalent to GenAI-Perf OSL, as it excludes reasoning tokens from the count
- **AIPerf Reasoning Token Count** is exclusive to AIPerf, as it includes reasoning tokens only

By providing OSL, Reasoning Token Count, and Output Token Count metrics, AIPerf enables more comprehensive performance analysis of reasoning-capable models by providing a complete picture of the token generation process.

