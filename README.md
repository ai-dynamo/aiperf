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


AIPerf is a comprehensive benchmarking tool for measuring the performance of generative AI models served your preferred inference solution. 
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
- Several benchmarking modes: concurrency, request-rate, trace replay

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
  --model-names your_model_name \
  --url http://localhost:8000 \
  --endpoint-type openai
```

### Example with Custom Configuration

```bash
aiperf profile \
  --model-names gpt-3.5-turbo \
  --url http://localhost:8000 \
  --endpoint-type openai \
  --concurrency 10 \
  --request-count 100 \
  --streaming
```

Example output:
<div align="center">

```
                              NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                         Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│          Time to first token (ms) │  16.26 │  12.39 │  17.25 │  17.09 │  16.68 │  16.56 │
│          Inter token latency (ms) │   1.85 │   1.55 │   2.04 │   2.02 │   1.97 │   1.92 │
│              Request latency (ms) │ 499.20 │ 451.01 │ 554.61 │ 548.69 │ 526.13 │ 514.19 │
│            Output sequence length │ 261.90 │ 256.00 │ 298.00 │ 296.60 │ 270.00 │ 265.00 │
│             Input sequence length │ 550.06 │ 550.00 │ 553.00 │ 551.60 │ 550.00 │ 550.00 │
│ Output token throughput (per sec) │ 520.87 │    N/A │    N/A │    N/A │    N/A │    N/A │
│      Request throughput (per sec) │   1.99 │    N/A │    N/A │    N/A │    N/A │    N/A │
└───────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```
</div>

</br>

