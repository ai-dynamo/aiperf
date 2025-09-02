<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Migrating from GenAI-Perf

AIPerf is designed to be a drop in replacement for [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer). There are only a few options from GenAI-Perf that do not directly map to AIPerf options. These are noted below.
Some options, mainly around the analyze subcommand, may not yet be supported but are coming in the near future.
<br>

## Known CLI Argument Differences

- `--max-threads`: Setting a max-thread option is no longer necessary. This was a global setting for GenAI-Perf controlling the total thread count.
In AIPerf, for more fine grained control over the number of workers issuing requests to the endpoint, the option `--workers-max` is available.
- `--`: The passthrough args flag is no longer required. All options are now natively supported by AIPerf.

Removing the above options should be all that is required to have your previous GenAI-Perf commands work in AIPerf.

<br>


---

With these simple updates to your previous scripts, AIPerf can replace your usage of GenAI-Perf.