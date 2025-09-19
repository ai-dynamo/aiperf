<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# GenAI-Perf vs AIPerf CLI Feature Comparison Matrix

This comprehensive comparison matrix shows the supported CLI options between GenAI-Perf and AIPerf tools.

**Legend:**
- ✅ **Fully Supported** - Feature available with same/similar functionality
- 🟡 **Partial Support** - Feature available but with different parameters or limitations
- **`N/A`** **Not Applicable** - Feature not applicable
- ❌ **Not Supported** - Feature not currently supported

---

## **Core Subcommands**

| Subcommand | Description | GenAI-Perf | AIPerf | Notes |
|------------|-------------|------------|---------|-------|
| **profile** | Profile LLMs and GenAI models | ✅ | ✅ | |
| **analyze** | Sweep through multiple scenarios | ✅ | ❌ | |
| **config** | Run using YAML configuration files | ✅ | ❌ | |
| **create-template** | Generate template configs | ✅ | ❌ | |
| **process-export-files** | Multi-node result aggregation | ✅ | **`N/A`** | Not applicable to AIPerf |

---

## **Endpoint Types Support Matrix**

`--endpoint-type`

| Endpoint Type | Description | GenAI-Perf | AIPerf | Notes |
|---------------|-------------|------------|---------|-------|
| **chat** | Standard chat completion API (OpenAI-compatible) | ✅ | ✅ | |
| **completions** | Text completion API for prompt completion | ✅ | ✅ | |
| **embeddings** | Text embedding generation for similarity/search | ✅ | ✅ | |
| **rankings** | Text ranking/reranking for search relevance | ✅ | ✅ | |
| **responses** | OpenAI responses endpoint | ❌ | 🟡 | Not fully supported yet |
| **dynamic_grpc** | Dynamic gRPC service calls | ✅ | ❌ | |
| **huggingface_generate** | HuggingFace transformers generate API | ✅ | ❌ | |
| **image_retrieval** | Image search and retrieval endpoints | ✅ | ❌ | |
| **nvclip** | NVIDIA CLIP model endpoints | ✅ | ❌ | |
| **multimodal** | Multi-modal (text + image/audio) endpoints | ✅ | 🟡 | use `chat` for AIPerf instead |
| **generate** | Generic text generation endpoints | ✅ | ❌ | |
| **kserve** | KServe model serving endpoints | ✅ | ❌ | |
| **template** | Template-based inference endpoints | ✅ | ❌ | |
| **tensorrtllm_engine** | TensorRT-LLM engine direct access | ✅ | ❌ | |
| **vision** | Computer vision model endpoints | ✅ | ❌ | |

---

## **Endpoint Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Model Names** | `-m` | ✅ | ✅ | |
| **Model Selection Strategy** | `--model-selection-strategy`<br>`{round_robin,random}` | ✅ | ✅ | |
| **Backend Selection** | `--backend`<br>`{tensorrtllm,vllm}` | ✅ | ❌ | |
| **Custom Endpoint** | `--endpoint` | ✅ | ✅ | |
| **Endpoint Type** | `--endpoint-type` | ✅ | ✅ | [See detailed comparison above](#endpoint-types-support-matrix) |
| **Server Metrics URL** | `--server-metrics-url` | ✅ | ❌ | |
| **Streaming** | `--streaming` | ✅ | ✅ | |
| **URL** | `-u URL`<br>`--url` | ✅ | ✅ | |
| **Request Timeout** | `--request-timeout-seconds` | ❌ | ✅ | |
| **API Key** | `--api-key` | 🟡 | ✅ | For GenAI-Perf, use `-H` instead |

---

## **Input Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Extra Inputs** | `--extra-inputs` | ✅ | ✅ | |
| **Custom Headers** | `--header -H` | ✅ | ✅ | |
| **Input File** | `--input-file` | ✅ | ✅ | |
| **Dataset Entries/Conversations** | `--num-dataset-entries` | ✅ | ✅ | |
| **Public Dataset** | `--public-dataset`<br>`{sharegpt}` | ❌ | ✅ | |
| **Custom Dataset Type** | `--custom-dataset-type`<br>`{single-turn,multi-turn,random-pool,mooncake-trace}` | 🟡 | ✅ | GenAI-Perf infers this from the input file |
| **Fixed Schedule** | `--fixed-schedule` | ✅ | ✅ | |
| **Fixed Schedule Auto Offset** | `--fixed-schedule-auto-offset` | ❌ | ✅ | |
| **Fixed Schedule Start/End Offset** | `--fixed-schedule-start-offset`<br>`--fixed-schedule-end-offset` | ❌ | ✅ | |
| **Random Seed** | `--random-seed` | ✅ | ✅ | |
| **GRPC Method** | `--grpc-method` | ✅ | ❌ | |

---

## **Output Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Artifact Directory** | `--artifact-dir` | ✅ | ✅ | |
| **Checkpoint Directory** | `--checkpoint-dir` | ✅ | ❌ | |
| **Generate Plots** | `--generate-plots` | ✅ | ❌ | |
| **Enable Checkpointing** | `--enable-checkpointing` | ✅ | ❌ | |
| **Profile Export File** | `--profile-export-file` | ✅ | ❌ | |

---

## **Tokenizer Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Tokenizer** | `--tokenizer` | ✅ | ✅ | |
| **Tokenizer Revision** | `--tokenizer-revision` | ✅ | ✅ | |
| **Tokenizer Trust Remote Code** | `--tokenizer-trust-remote-code` | ✅ | ✅ | |

---

## **Load Generator Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Concurrency** | `--concurrency` | ✅ | ✅ | |
| **Request Rate** | `--request-rate` | ✅ | ✅ | |
| **Request Rate Mode** | `--request-rate-mode`<br>`{constant,poisson}` | ❌ | ✅ | |
| **Request Rate w/ Max Concurrency** | `--request-rate` with `--concurrency` | ❌ | ✅ | |
| **Request Count** | `--request-count`<br>`--num-requests` | ✅ | ✅ | |
| **Measurement Interval** | `--measurement-interval -p` | ✅ | **`N/A`** | Not applicable to AIPerf |
| **Stability Percentage** | `--stability-percentage -s` | ✅ | **`N/A`** | Not applicable to AIPerf |
| **Warmup Request Count** | `--warmup-request-count`<br>`--num-warmup-requests` | ✅ | ✅ | |

---

## **Session/Conversation Configuration (Multi-turn)**

> [!NOTE]
> AIPerf does not currently support benchmarking with multiple turns/sessions. The following options only apply to the generation of synthetic data.


| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Number of Sessions** | `--num-sessions` | ✅ | 🟡 | |
| **Session Concurrency** | `--session-concurrency` | ✅ | ❌ | |
| **Session Delay Ratio** | `--session-delay-ratio` | ✅ | 🟡 | |
| **Session Turn Delay Mean** | `--session-turn-delay-mean` | ✅ | 🟡 | |
| **Session Turn Delay Stddev** | `--session-turn-delay-stddev` | ✅ | 🟡 | |
| **Session Turns Mean** | `--session-turns-mean` | ✅ | 🟡 | |
| **Session Turns Stddev** | `--session-turns-stddev` | ✅ | 🟡 | |

---

## **Input Sequence Length (ISL) Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Input Tokens Mean** | `--synthetic-input-tokens-mean`<br>`--isl` | ✅ | ✅ | |
| **Input Tokens Stddev** | `--synthetic-input-tokens-stddev` | ✅ | ✅ | |
| **Input Tokens Block Size** | `--prompt-input-tokens-block-size`<br>`--isl-block-size` | ❌ | ✅ | |

---

## **Output Sequence Length (OSL) Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Output Tokens Mean** | `--output-tokens-mean`<br>`--osl` | ✅ | ✅ | |
| **Output Tokens Stddev** | `--output-tokens-stddev` | ✅ | ✅ | |
| **Output Tokens Mean Deterministic** | `--output-tokens-mean-deterministic` | ✅ | ❌ | Only applicable to Triton |

---

## **Batch Size Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Text Batch Size** | `--batch-size-text`<br>`--batch-size -b` | ✅ | ✅ | |
| **Audio Batch Size** | `--batch-size-audio` | ✅ | ✅ | |
| **Image Batch Size** | `--batch-size-image` | ✅ | ✅ | |

---

## **Prefix Prompt Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Number of Prefix Prompts** | `--num-prefix-prompts` | ✅ | ✅ | |
| **Prefix Prompt Length** | `--prefix-prompt-length` | ✅ | ✅ | |

---

## **Audio Input Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Audio Length Mean** | `--audio-length-mean` | ✅ | ✅ | |
| **Audio Length Stddev** | `--audio-length-stddev` | ✅ | ✅ | |
| **Audio Format** | `--audio-format`<br>`{wav,mp3,random}` | ✅ | ✅ | |
| **Audio Depths** | `--audio-depths` | ✅ | ✅ | |
| **Audio Sample Rates** | `--audio-sample-rates` | ✅ | ✅ | |
| **Audio Number of Channels** | `--audio-num-channels` | ✅ | ✅ | |

---

## **Image Input Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Image Width Mean** | `--image-width-mean` | ✅ | ✅ | |
| **Image Width Stddev** | `--image-width-stddev` | ✅ | ✅ | |
| **Image Height Mean** | `--image-height-mean` | ✅ | ✅ | |
| **Image Height Stddev** | `--image-height-stddev` | ✅ | ✅ | |
| **Image Format** | `--image-format`<br>`{png,jpeg,random}` | ✅ | ✅ | |

---

## **Service Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Record Processor Service Count** | `--record-processor-service-count`<br>`--record-processors` | ❌ | ✅ | |
| **Maximum Workers** | `--workers-max`<br>`--max-workers` | ❌ | ✅ | |
| **ZMQ Host** | `--zmq-host` | ❌ | ✅ | |
| **ZMQ IPC Path** | `--zmq-ipc-path` | ❌ | ✅ | |

---

## **Additional Features**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Goodput Constraints** | `--goodput -g` | ✅ | ❌ | |
| **Verbose** | `-v --verbose` | ✅ | ✅ | |
| **Extra Verbose** | `-vv` | ✅ | ✅ | |
| **Log Level** | `--log-level` | ❌ | ✅ | `{trace,debug,info,notice,warning,success,error,critical}` |
| **UI Type** | `--ui-type --ui`<br>`{dashboard,simple,none}` | ❌ | ✅ | |
| **Help** | `-h --help` | ✅ | ✅ | |

---

## **Perf-Analyzer Passthrough Arguments**

> [!NOTE]
> GenAI-Perf supports passing through arguments to the Perf-Analyzer CLI. AIPerf does not support this, as it does not use Perf-Analyzer under the hood.

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Perf-Analyzer Passthrough Arguments** | `--` | ✅ | **`N/A`** | Only applicable to GenAI-Perf |


---

## **Data Exporters**

| Feature | GenAI-Perf | AIPerf | Notes |
|---------|------------|--------|-------|
| Console output | ✅ | ✅ | |
| JSON output | ✅ | ✅ | |
| CSV output | ✅ | ✅ |  |
| API Error Summary | ❌ | ✅ | |
| `profile_export.json` | ✅ | ❌ | |
| `inputs.json` | ✅ | ❌ | |

---

## **Advanced Features Comparison**

| Feature | GenAI-Perf | AIPerf | Notes |
|---------|------------|--------|-------|
| **Multi-modal support** | ✅ | 🟡 | |
| **GPU Telemetry** | ✅ | ❌ | |
| **Streaming API support** | ✅ | ✅ | |
| **Multi-turn conversations** | ✅ | ❌ |  |
| **Payload scheduling** | ✅ | ✅ | Fixed schedule workloads |
| **Distributed testing** | ✅ | ❌ | Multi-node result aggregation |
| **Custom endpoints** | ✅ | ✅ |  |
| **Synthetic data generation** | ✅ | ✅ | |
| **Bring Your Own Data (BYOD)** | ✅ | ✅ | Custom dataset support |
| **Audio metrics** | ✅ | ❌ | Audio-specific performance metrics |
| **Vision metrics** | ✅ | ❌ | Image-specific performance metrics |
| **Live Metrics** | ❌ | ✅ | Live metrics display |
| **Dashboard UI** | ❌ | ✅ | Dashboard UI |
| **Reasoning token parsing** | ❌ | ✅ | Parsing of reasoning tokens |

---
