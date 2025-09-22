<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Trace Replay

This tutorial takes you through an example trace replay profile. Trace Replay benchmarking helps reproduce performance benchmarking results for validation or testing your system under a specific load pattern.

## Table of Contents

- [MoonCake Traces](#mooncake-traces)
- [Profiling using a MoonCake Trace](#profiling-using-a-mooncake-trace)
- [Real Mooncake Trace Example](#real-mooncake-trace-example)
  - [Download and Benchmark Mooncake Trace](#download-and-benchmark-mooncake-trace)

## MoonCake Traces

Mooncake provides a specification and sample datasets for [traces](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#-open-source-trace) that can be replayed for performance benchmarking.

In AIPerf, the trace must be defined in a jsonl file.

4 keys are available:
- "timestamp": the timing of request arrivals
- "input_length": the number of input tokens
- "output_length": the number of output tokens
- "hash_ids": [list of block hashes]



```json
{
    "timestamp": 0,
    "input_length": 655,
    "output_length": 52,
    "hash_ids": [46, 47]
}
```



## Profiling using a MoonCake Trace


```bash
echo \
'{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
{"timestamp": 10535, "input_length": 672, "output_length": 26, "hash_ids": [46, 47]}
{"timestamp": 27482, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}' \
> example_trace.jsonl

aiperf profile \
    -m deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
	--input-file example_trace.jsonl \
	--custom-dataset-type mooncake_trace \
    --fixed-schedule
```

The code above will create an example trace file formatted in jsonl. AIPerf will use it to define the dataset and timing to replay the trace.

## Real Mooncake Trace Example

For real-world benchmarking, you can use the actual FAST25 production trace data from the Mooncake research paper. This trace contains realistic request patterns from production workloads.

### Download and Benchmark Mooncake Trace

```bash
# Download the Mooncake trace data
curl -o mooncake_trace.jsonl https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl

# Create a subset of the file for quick testing (skip if you want to run the full dataset)
head -n 5 mooncake_trace.jsonl > mooncake_trace_short.jsonl

# Run a small portion of the trace dataset
aiperf profile \
    -m deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --input-file mooncake_trace_short.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule
```
