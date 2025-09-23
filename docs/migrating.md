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


---

With these simple updates to your previous scripts, AIPerf can replace your usage of GenAI-Perf.
