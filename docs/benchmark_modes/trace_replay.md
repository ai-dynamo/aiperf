<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Trace Replay

This turorial takes you through an example trace replay profile. Trace Replay benchmarking helps reproduce performance benchmarking results for validation or testing your system under a specific load pattern.

## Table of Contents

- [MoonCake Traces](#mooncake-traces)
- [Profiling using a MoonCake Trace](#profiling-using-a-mooncake-trace)

## MoonCake Traces

Mooncake provides a specification and sample datasets for [traces](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#-open-source-trace) that can be replayed for performance benchmarking. 

In AIPerf, the trace must be defined in a jsonl file. 

5 keys are available:
- "timestamp": the timing of request arrivals
- "input_length": the number of input tokens
- "output_length": the number of output tokens
- "hash_ids": [list of block hashes]
- "text": the text prompt tokenize for your input



Several combinations are supported.


```json
{
    "timestamp": 0, 
    "input_length": 655,
    "output_length": 52,
    "hash_ids": [46, 47]
}
```

```json
{
    "timestamp": 0, 
    "input_length": 655,
    "hash_ids": [46, 47]
}
```

```json
{
    "timestamp": 0, 
    "text": "Hello, My name is"
}
```

```json
{
    "timestamp": 0, 
    "output_length": 52,
    "text": "Hello, My name is"
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
	--custom-dataset-type mooncake_trace
```

The code above will create an example trace file formatted in jsonl. AIPerf will use it to define the dataset and timing to replay the trace.