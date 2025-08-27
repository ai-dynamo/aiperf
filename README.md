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
# AIPerf

[![PyPI version](https://img.shields.io/pypi/v/AIPerf)](https://pypi.org/project/aiperf/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://dcbadge.limes.pink/api/server/D92uqZRjCZ?style=flat)](https://discord.gg/D92uqZRjCZ)

| **[Design Proposals](https://github.com/ai-dynamo/enhancements)** | **[Migrating from Genai-Perf](docs/migrating.md)** | **[CLI Options](docs/cli_options.md)**


AIPerf is a comprehensive benchmarking tool for measuring the performance of generative AI models served by your preferred inference solution.
It provides detailed metrics via a command line display as well as extensive benchmark performance reports.

AIPerf provides multiprocess and kubernetes support (coming soon) out of the box for a single scalable solution.

</br>

<!--
======================
Features
======================
-->

## Features

- Scalable via multiprocess or Kubernetes (coming soon) support
- Modular design for easy user modification
- Several benchmarking modes:
  - concurrency
  - request-rate
  - request-rate with a maximum concurrency
  - [trace replay](docs/benchmark_modes/trace_replay.md)

</br>

<!--
======================
INSTALLATION
======================
-->

## Installation
```
pip install git+https://github.com/ai-dynamo/aiperf.git
```

</br>

<!--
======================
QUICK START
======================
-->

## Quick Start

### Basic Usage

Run a simple benchmark against a model:

```bash
aiperf profile \
  --model your_model_name \
  --url http://localhost:8000 \
  --endpoint-type chat
  --streaming
```

### Example with Custom Configuration

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --concurrency 10 \
  --request-count 100 \
  --streaming
```

Example output:
<div align="center">

```
NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃                               Metric ┃       avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│             Time to First Token (ms) │     18.26 │  11.22 │ 106.32 │  68.82 │  27.76 │  16.62 │ 12.07 │
│            Time to Second Token (ms) │     11.40 │   0.02 │  85.91 │  34.54 │  12.59 │  11.65 │  7.01 │
│                 Request Latency (ms) │    487.30 │ 267.07 │ 769.57 │ 715.99 │ 580.83 │ 536.17 │ 79.60 │
│             Inter Token Latency (ms) │     11.23 │   8.80 │  13.17 │  12.48 │  11.73 │  11.37 │  0.45 │
│     Output Token Throughput Per User │     89.23 │  75.93 │ 113.60 │ 102.28 │  90.91 │  90.29 │  3.70 │
│                    (tokens/sec/user) │           │        │        │        │        │        │       │
│      Output Sequence Length (tokens) │     42.83 │  24.00 │  65.00 │  64.00 │  52.00 │  47.00 │  7.21 │
│       Input Sequence Length (tokens) │     10.00 │  10.00 │  10.00 │  10.00 │  10.00 │  10.00 │  0.00 │
│ Output Token Throughput (tokens/sec) │ 10,944.03 │    N/A │    N/A │    N/A │    N/A │    N/A │   N/A │
│    Request Throughput (requests/sec) │    255.54 │    N/A │    N/A │    N/A │    N/A │    N/A │   N/A │
│             Request Count (requests) │    711.00 │    N/A │    N/A │    N/A │    N/A │    N/A │   N/A │
└──────────────────────────────────────┴───────────┴────────┴────────┴────────┴────────┴────────┴───────┘
```
</div>


