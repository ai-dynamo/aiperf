---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: GenAI-Perf vs AIPerf CLI Feature Comparison Matrix
---
# GenAI-Perf vs AIPerf CLI Feature Comparison Matrix

This comparison matrix shows the supported CLI options between GenAI-Perf and AIPerf.

<Note>
This is a living document and will be updated as new features are added to AIPerf.
</Note>


**Legend:**
- ✅ **Fully Supported** - Feature available with same/similar functionality
- 🟡 **Partial Support** - Feature available but with different parameters or limitations
- **`N/A`** **Not Applicable** - Feature not applicable
- ❌ **Not Supported** - Feature not currently supported

---

## **Core Subcommands**

| Subcommand | Description | GenAI-Perf | AIPerf | Notes |
|------------|-------------|------------|---------|-------|
| **analyze-trace** | Analyze mooncake trace for prefix statistics | ❌ | ✅ | |
| **profile** | Profile LLMs and GenAI models | ✅ | ✅ | |
| **plot** | Generate visualizations from profiling data | ❌ | ✅ | Auto-detects multi-run comparison vs single-run analysis; supports dashboard mode |
| **analyze** | Sweep through multiple scenarios | ✅ | ❌ | |
| **config** | Run using YAML configuration files | ✅ | ❌ | |
| **create-template** | Generate template configs | ✅ | ❌ | |
| **process-export-files** | Multi-node result aggregation | ✅ | **`N/A`** | AIPerf will aggregate results in real-time |

---

## **Endpoint Types Support Matrix**

`--endpoint-type`

| Endpoint Type | Description | GenAI-Perf | AIPerf | Notes |
|---------------|-------------|------------|---------|-------|
| **chat** | Standard chat completion API (OpenAI-compatible) | ✅ | ✅ | |
| **completions** | Text completion API for prompt completion | ✅ | ✅ | |
| **embeddings** | Text embedding generation for similarity/search | ✅ | ✅ | |
| **rankings** | Text ranking/re-ranking for search relevance | ✅ | ✅ | GenAI-Perf's generic `rankings` is HF TEI compatible; AIPerf has separate `nim_rankings`, `hf_tei_rankings` and `cohere_rankings` |
| **hf_tei_rankings** | HuggingFace TEI re-ranker API | ✅ | ✅ | GenAI-Perf uses generic `rankings` endpoint |
| **nim_rankings** | NVIDIA NIM re-ranker API | ❌ | ✅ | |
| **cohere_rankings** | Cohere re-ranker API | ❌ | ✅ | |
| **responses** | OpenAI responses endpoint | ❌ | ❌ | |
| **dynamic_grpc** | Dynamic gRPC service calls | ✅ | ❌ | |
| **huggingface_generate** | HuggingFace transformers generate API | ✅ | ✅ | `/generate` and `/generate_stream` supported |
| **image_generation** | OpenAI-compatible image generation (`/v1/images/generations`) | ❌ | ✅ | Text-to-image benchmarking with SGLang, supports raw export for image extraction |
| **image_retrieval** | Image search and retrieval endpoints | ✅ | ❌ | |
| **nvclip** | NVIDIA CLIP model endpoints | ✅ | ❌ | |
| **multimodal** | Multi-modal (text + image/audio) endpoints | ✅ | ✅ | AIPerf uses `chat` endpoint with multimodal content |
| **generate** | Generic text generation endpoints | ✅ | ❌ | |
| **kserve** | KServe model serving endpoints | ✅ | ❌ | |
| **template** | Template-based inference endpoints | 🟡 | ✅ | AIPerf supports multimodal and multi-turn templates |
| **tensorrtllm_engine** | TensorRT-LLM engine direct access | ✅ | ❌ | |
| **vision** | Computer vision model endpoints | ✅ | ✅ | AIPerf uses `chat` endpoint for VLMs |
| **solido_rag** | SOLIDO RAG endpoint | ❌ | ✅ | |

---

## **Endpoint Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Model Names** | `-m` | ✅ | ✅ | |
| **Model Selection Strategy** | `--model-selection-strategy`<br/>`{round_robin,random}` | ✅ | ✅ | |
| **Backend Selection** | `--backend`<br/>`{tensorrtllm,vllm}` | ✅ | ❌ | |
| **Custom Endpoint** | `--endpoint` | ✅ | ✅ | |
| **Endpoint Type** | `--endpoint-type` | ✅ | ✅ | [See detailed comparison above](#endpoint-types-support-matrix) |
| **Server Metrics URL** | `--server-metrics-url` | ❌ | ✅ | AIPerf uses `--server-metrics` (enabled by default, auto-collects Prometheus metrics from endpoint). GenAI-Perf's `--server-metrics-url` is for GPU telemetry only. |
| **Streaming** | `--streaming` | ✅ | ✅ | |
| **URL** | `-u URL`<br/>`--url` | ✅ | ✅ | |
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
| **Public Dataset** | `--public-dataset`<br/>`{sharegpt}` | ❌ | ✅ | |
| **Custom Dataset Type** | `--custom-dataset-type`<br/>`{single_turn,multi_turn,random_pool,mooncake_trace}` | ❌ | ✅ | GenAI-Perf infers dataset type from input file format |
| **Fixed Schedule** | `--fixed-schedule` | ✅ | ✅ | |
| **Fixed Schedule Auto Offset** | `--fixed-schedule-auto-offset` | ❌ | ✅ | |
| **Fixed Schedule Start/End Offset** | `--fixed-schedule-start-offset`<br/>`--fixed-schedule-end-offset` | ❌ | ✅ | |
| **Random Seed** | `--random-seed` | ✅ | ✅ | |
| **GRPC Method** | `--grpc-method` | ✅ | ❌ | |

---

## **Output Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Artifact Directory** | `--artifact-dir` | ✅ | ✅ | |
| **Checkpoint Directory** | `--checkpoint-dir` | ✅ | ❌ | |
| **Generate Plots** | `--generate-plots` | ✅ | 🟡 | AIPerf uses separate `aiperf plot` subcommand with more features |
| **Enable Checkpointing** | `--enable-checkpointing` | ✅ | ❌ | |
| **Profile Export File** | `--profile-export-file` | ✅ | ✅ | AIPerf works as a prefix for the profile export file names. |

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
| **Request Count** | `--request-count`<br/>`--num-requests` | ✅ | ✅ | |
| **Request Rate w/ Max Concurrency** | `--request-rate` with `--concurrency` | ❌ | ✅ | Dual control of rate and concurrency ceiling |
| **Measurement Interval** | `--measurement-interval -p` | ✅ | **`N/A`** | Not applicable to AIPerf |
| **Stability Percentage** | `--stability-percentage -s` | ✅ | **`N/A`** | Not applicable to AIPerf |

---

## **Arrival Pattern Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Arrival Pattern** | `--arrival-pattern`<br/>`{constant,poisson,gamma}` | ❌ | ✅ | Controls inter-arrival time distribution |
| **Arrival Smoothness** | `--arrival-smoothness`<br/>`--vllm-burstiness` | ❌ | ✅ | Gamma distribution shape: &lt;1=bursty, 1=Poisson, >1=smooth |

---

## **Duration-Based Benchmarking**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Benchmark Duration** | `--benchmark-duration` | ❌ | ✅ | Stop after N seconds |
| **Benchmark Grace Period** | `--benchmark-grace-period` | ❌ | ✅ | Wait for in-flight requests after duration (default: 30s, supports `inf`) |

---

## **Concurrency Control**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Session Concurrency** | `--concurrency` | ✅ | ✅ | Max concurrent sessions |
| **Prefill Concurrency** | `--prefill-concurrency` | ❌ | ✅ | Limit concurrent prefill operations (requires `--streaming`) |

---

## **Gradual Ramping**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Concurrency Ramp** | `--concurrency-ramp-duration` | ❌ | ✅ | Ramp concurrency from 1 to target over N seconds |
| **Prefill Concurrency Ramp** | `--prefill-concurrency-ramp-duration` | ❌ | ✅ | Ramp prefill concurrency over N seconds |
| **Request Rate Ramp** | `--request-rate-ramp-duration` | ❌ | ✅ | Ramp request rate over N seconds |

---

## **User-Centric Timing (KV Cache Benchmarking)**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **User-Centric Rate** | `--user-centric-rate` | ❌ | ✅ | Per-user rate limiting with consistent turn gaps |
| **Number of Users** | `--num-users` | ❌ | ✅ | Number of simulated users (required with `--user-centric-rate`) |
| **Shared System Prompt** | `--shared-system-prompt-length` | ❌ | ✅ | System prompt shared across all users (KV cache prefix) |
| **User Context Prompt** | `--user-context-prompt-length` | ❌ | ✅ | Per-user unique context padding |

---

## **Warmup Phase Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Warmup Request Count** | `--warmup-request-count` | ✅ | ✅ | |
| **Warmup Duration** | `--warmup-duration` | ❌ | ✅ | Duration-based warmup stop condition |
| **Warmup Session Count** | `--num-warmup-sessions` | ❌ | ✅ | Session-based warmup stop condition |
| **Warmup Concurrency** | `--warmup-concurrency` | ❌ | ✅ | Override concurrency during warmup |
| **Warmup Prefill Concurrency** | `--warmup-prefill-concurrency` | ❌ | ✅ | Override prefill concurrency during warmup |
| **Warmup Request Rate** | `--warmup-request-rate` | ❌ | ✅ | Override request rate during warmup |
| **Warmup Arrival Pattern** | `--warmup-arrival-pattern` | ❌ | ✅ | Override arrival pattern during warmup |
| **Warmup Grace Period** | `--warmup-grace-period` | ❌ | ✅ | Grace period for warmup responses |
| **Warmup Concurrency Ramp** | `--warmup-concurrency-ramp-duration` | ❌ | ✅ | Ramp warmup concurrency |
| **Warmup Prefill Ramp** | `--warmup-prefill-concurrency-ramp-duration` | ❌ | ✅ | Ramp warmup prefill concurrency |
| **Warmup Rate Ramp** | `--warmup-request-rate-ramp-duration` | ❌ | ✅ | Ramp warmup request rate |

---

## **Session/Conversation Configuration (Multi-turn)**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Number of Sessions** | `--num-sessions` | ✅ | ✅ | |
| **Session Concurrency** | `--session-concurrency` | ✅ | ✅ | Use `--concurrency` for AIPerf |
| **Session Delay Ratio** | `--session-delay-ratio` | ✅ | ✅ | |
| **Session Turn Delay Mean** | `--session-turn-delay-mean` | ✅ | ✅ | |
| **Session Turn Delay Stddev** | `--session-turn-delay-stddev` | ✅ | ✅ | |
| **Session Turns Mean** | `--session-turns-mean` | ✅ | ✅ | |
| **Session Turns Stddev** | `--session-turns-stddev` | ✅ | ✅ | |

---

## **Input Sequence Length (ISL) Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Input Tokens Mean** | `--synthetic-input-tokens-mean`<br/>`--isl` | ✅ | ✅ | |
| **Input Tokens Stddev** | `--synthetic-input-tokens-stddev` | ✅ | ✅ | |
| **Input Tokens Block Size** | `--prompt-input-tokens-block-size`<br/>`--isl-block-size` | ❌ | ✅ | Used for `mooncake_trace` hash_id blocks |

---

## **Output Sequence Length (OSL) Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Output Tokens Mean** | `--output-tokens-mean`<br/>`--osl` | ✅ | ✅ | |
| **Output Tokens Stddev** | `--output-tokens-stddev` | ✅ | ✅ | |
| **Output Tokens Mean Deterministic** | `--output-tokens-mean-deterministic` | ✅ | ❌ | Only applicable to Triton |

---

## **Batch Size Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Text Batch Size** | `--batch-size-text`<br/>`--batch-size -b` | ✅ | ✅ | |
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
| **Audio Format** | `--audio-format`<br/>`{wav,mp3,random}` | ✅ | ✅ | |
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
| **Image Format** | `--image-format`<br/>`{png,jpeg,random}` | ✅ | ✅ | |

---

## **Service Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Record Processor Service Count** | `--record-processor-service-count`<br/>`--record-processors` | ❌ | ✅ | |
| **Maximum Workers** | `--workers-max`<br/>`--max-workers` | ❌ | ✅ | |
| **ZMQ Host** | `--zmq-host` | ❌ | ✅ | |
| **ZMQ IPC Path** | `--zmq-ipc-path` | ❌ | ✅ | |

---

## **Request Cancellation**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Request Cancellation Rate** | `--request-cancellation-rate` | ❌ | ✅ | Percentage of requests to cancel (0-100) |
| **Request Cancellation Delay** | `--request-cancellation-delay` | ❌ | ✅ | Seconds to wait before cancelling |

---

## **Additional Features**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Goodput Constraints** | `--goodput -g` | ✅ | ✅ | |
| **Verbose** | `-v --verbose` | ✅ | ✅ | |
| **Extra Verbose** | `-vv` | ✅ | ✅ | |
| **Log Level** | `--log-level` | ❌ | ✅ | `{trace,debug,info,notice,warning,success,error,critical}` |
| **UI Type** | `--ui-type --ui`<br/>`{dashboard,simple,none}` | ❌ | ✅ | |
| **Help** | `-h --help` | ✅ | ✅ | |

---

## **Perf-Analyzer Passthrough Arguments**

<Note>
GenAI-Perf supports passing through arguments to the Perf-Analyzer CLI. AIPerf does not support this, as it does not use Perf-Analyzer under the hood.
</Note>

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Perf-Analyzer Passthrough Arguments** | `--` | ✅ | **`N/A`** | Only applicable to GenAI-Perf |


---

## **Data Exporters**

| Feature | GenAI-Perf | AIPerf | Notes |
|---------|------------|--------|-------|
| Console output | ✅ | ✅ | |
| JSON output | ✅ | ✅ | [See discrepancies below](#json-output) |
| CSV output | ✅ | ✅ | |
| API Error Summary | ❌ | ✅ | |
| `profile_export.json` | ✅ | ✅ | Use `--export-level raw` in AIPerf to get raw input/output payloads |
| Per-Record Metrics | ❌ | ✅ | |
| `inputs.json` | ✅ | ✅ | AIPerf format is slightly different |

### Discrepancies

#### JSON Output

- Fields in the `input_config` section may differ between GenAI-Perf and AIPerf.

---

## **Advanced Features Comparison**

| Feature | GenAI-Perf | AIPerf | Notes |
|---------|------------|--------|-------|
| **Multi-modal support** | ✅ | ✅ | |
| **GPU Telemetry** | ✅ | ✅ | |
| **Streaming API support** | ✅ | ✅ | |
| **Multi-turn conversations** | ✅ | ✅ | Full multi-turn benchmarking with session tracking |
| **Payload scheduling** | ✅ | ✅ | Fixed schedule workloads |
| **Distributed testing** | ✅ | 🟡 | Multi-node result aggregation |
| **Custom endpoints** | ✅ | ✅ |  |
| **Synthetic data generation** | ✅ | ✅ | |
| **Bring Your Own Data (BYOD)** | ✅ | ✅ | Custom dataset support |
| **Audio metrics** | ✅ | ❌ | Audio-specific performance metrics |
| **Vision metrics** | ✅ | ✅ | Image-specific performance metrics |
| **Image generation benchmarking** | ❌ | ✅ | Text-to-image with raw export for image extraction |
| **Live Metrics** | ❌ | ✅ | Live metrics display |
| **Dashboard UI** | ❌ | ✅ | Dashboard UI |
| **Reasoning token parsing** | ❌ | ✅ | Parsing of reasoning tokens |
| **Arrival pattern control** | ❌ | ✅ | Constant, Poisson, Gamma distributions with tunable burstiness |
| **Prefill concurrency limiting** | ❌ | ✅ | Fine-grained prefill queueing control for TTFT behavior |
| **Gradual ramping** | ❌ | ✅ | Smooth ramp-up for concurrency and rate |
| **Duration-based benchmarking** | ❌ | ✅ | Time-based stop conditions with grace periods |
| **User-centric timing** | ❌ | ✅ | Per-user rate limiting for KV cache benchmarking |
| **Configurable warmup phase** | 🟡 | ✅ | AIPerf supports full warmup configuration (rate, concurrency, duration, ramping) |
| **HTTP trace metrics** | ❌ | ✅ | Detailed HTTP lifecycle timing (DNS, TCP, TLS, TTFB) |
| **Request cancellation** | ❌ | ✅ | Test timeout behavior and service resilience |
| **Timeslice metrics** | ❌ | ✅ | Per-timeslice metric breakdown |
| **Interactive plot dashboard** | ❌ | ✅ | Web-based exploration with dynamic metric selection and filtering |
| **Multi-run comparison plots** | ❌ | ✅ | Auto-detected Pareto curves and throughput analysis |

---
