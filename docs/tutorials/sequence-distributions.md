<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sequence Length Distributions for Advanced Benchmarking

Sequence length distributions allow you to specify complex patterns of input
sequence length (ISL) and output sequence length (OSL) pairs with different
probabilities and optional variance. This enables benchmarking of multiple
use cases such as summarization and Q&A on one endpoint.

## Overview

The sequence distribution feature provides benchmarking of mixed workloads
with different ISL and OSL pairings.

## Basic Usage

### Example command

Add variance to make workloads more realistic:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --sequence-distribution "64|10,32|8:70;256|40,128|20:20;1024|100,512|50:10" \
```

This creates:
- 70% of requests with ISL ~ Normal(64, 10), OSL ~ Normal(32, 8)
- 20% of requests with ISL ~ Normal(256, 40), OSL ~ Normal(128, 20)
- 10% of requests with ISL ~ Normal(1024, 100), OSL ~ Normal(512, 50)

Values are automatically clamped to be at least 1.

## Supported Formats

### 1. Semicolon Format (Recommended)

**Basic:**
```
"ISL1,OSL1:PROB1;ISL2,OSL2:PROB2;..."
```

**With standard deviations:**
```
"ISL1|STDDEV1,OSL1|STDDEV1:PROB1;ISL2|STDDEV2,OSL2|STDDEV2:PROB2"
```

### 2. Bracket Format

**Basic:**
```
"[(ISL1,OSL1):PROB1,(ISL2,OSL2):PROB2]"
```

**With standard deviations:**
```
"[(256|10,128|5):60,(512|20,256|15):40]"
```

### 3. JSON Format

**Basic:**
```json
{"pairs": [{"isl": 256, "osl": 128, "prob": 60}, {"isl": 512, "osl": 256, "prob": 40}]}
```

**With standard deviations:**
```json
{"pairs": [
  {"isl": 256, "isl_stddev": 10, "osl": 128, "osl_stddev": 5, "prob": 60},
  {"isl": 512, "isl_stddev": 20, "osl": 256, "osl_stddev": 15, "prob": 40}
]}
```

## Examples

### Example Case: Chatbot Workload Simulation

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Simulate typical chatbot traffic:
# - 70% short queries (quick questions)
# - 20% medium queries (explanations)
# - 10% long queries (complex tasks)

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --sequence-distribution "64|10,32|8:70;256|40,128|20:20;1024|100,512|50:10"
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

