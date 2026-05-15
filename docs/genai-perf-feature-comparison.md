---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: GenAI-Perf vs AIPerf CLI Feature Comparison Matrix
---
# GenAI-Perf vs AIPerf CLI Feature Comparison Matrix

This comparison matrix shows the supported CLI options between GenAI-Perf and AIPerf.

> [!NOTE]
> This is a living document and will be updated as new features are added to AIPerf.


**Legend:**
- ✅ **Fully Supported** - Feature available with same/similar functionality
- ⭐ **Enhanced** - Feature available in both tools, with broader capabilities or better ergonomics on the marked side
- 🟡 **Partial Support** - Feature available but with different parameters or limitations
- **`N/A`** **Not Applicable** - Feature not applicable
- ❌ **Not Supported** - Feature not currently supported

> [!NOTE]
> AIPerf is the successor to GenAI-Perf, so most ⭐ marks fall in AIPerf's column. They flag rows where AIPerf doesn't merely match the GenAI-Perf surface but expands on it. ❌ vs ✅ rows are already self-explanatory and are left unannotated.

---

## **Core Subcommands**

| Subcommand | Description | GenAI-Perf | AIPerf | Notes |
|------------|-------------|------------|---------|-------|
| **analyze-trace** | Analyze mooncake trace for prefix statistics | ❌ | ✅ | |
| **profile** | Profile LLMs and GenAI models | ✅ | ✅ | AIPerf accepts a YAML config via `profile -f config.yaml` (CLI flags override) |
| **plot** | Generate visualizations from profiling data | ❌ | ✅ | Auto-detects multi-run comparison vs single-run analysis; renders Pareto overlays when multiple artifact dirs are passed; supports dashboard mode |
| **analyze** | Sweep through multiple scenarios | ✅ | ✅ | AIPerf folds sweeps into `profile` via magic-list CLI flags, `--variant`, or YAML `sweep:` blocks (grid, zip, Sobol, Latin Hypercube). See [Parameter Sweeping](#parameter-sweeping) |
| **config** | Run a YAML config end-to-end | ✅ (separate `config` subcommand) | 🟡 | AIPerf has no `aiperf config <yaml>` run shortcut — pass `-f config.yaml` to `aiperf profile` instead |
| **create-template / config init** | Scaffold a template config | ✅ (GenAI-Perf: `create-template`) | ✅ | AIPerf: `aiperf config init -t <template>`; supports `--list`, `--search`, `--category` for discovery |
| **config expand** | Preview a sweep without running it | ❌ | ✅ | Prints every variation the orchestrator would iterate; `--full`/`--index`/`--format` controls verbosity |
| **config validate** | Pre-flight validate a config file | ❌ | ✅ | Runs the same load pipeline as `profile`; non-zero exit on fatal errors, warnings to stderr |
| **plugins** | List/inspect registered plugins | ❌ | ✅ | `aiperf plugins` enumerates planners, recipes, exporters, dataset loaders, and more |
| **synthesize** | Materialize a synthetic dataset to disk | ❌ | ✅ | Useful for caching dataset generation between repeated sweep cells |
| **process-export-files** | Multi-node result aggregation | ✅ | **`N/A`** | AIPerf aggregates results in real-time |

---

## **Endpoint Types Support Matrix**

`--endpoint-type`

| Endpoint Type | Description | GenAI-Perf | AIPerf | Notes |
|---------------|-------------|------------|---------|-------|
| **chat** | Standard chat completion API (OpenAI-compatible) | ✅ | ✅ | |
| **completions** | Text completion API for prompt completion | ✅ | ✅ | |
| **embeddings** | Text embedding generation for similarity/search | ✅ | ✅ | |
| **rankings** | Text ranking/re-ranking for search relevance | ✅ | ✅ ⭐ | GenAI-Perf has a single generic `rankings` endpoint (`/v1/ranking`, HF-TEI-compatible). AIPerf splits it into dedicated `nim_rankings`, `hf_tei_rankings`, and `cohere_rankings` endpoints. |
| **hf_tei_rankings** | HuggingFace TEI re-ranker API | 🟡 | ✅ | GenAI-Perf has only generic `rankings`; AIPerf has a dedicated endpoint at `/rerank` |
| **nim_rankings** | NVIDIA NIM re-ranker API | ❌ | ✅ | |
| **cohere_rankings** | Cohere re-ranker API | ❌ | ✅ | |
| **chat_embeddings** | Chat-style multimodal embeddings (vLLM VLM2Vec) | ❌ | ✅ | |
| **embeddings (NIM)** | NVIDIA NIM embeddings endpoint | ❌ | ✅ | AIPerf `nim_embeddings`; supports text and image inputs |
| **responses** | OpenAI Responses API endpoint | ❌ | ✅ | Multi-modal (text, image, audio) with streaming |
| **dynamic_grpc** | Dynamic gRPC service calls | ✅ | ❌ | |
| **huggingface_generate** | HuggingFace TGI generate API | ✅ | ✅ | `/generate` and `/generate_stream` supported |
| **image_generation** | OpenAI-compatible image generation (`/v1/images/generations`) | ❌ | ✅ | DALL-E-style text-to-image; supports raw export for image extraction |
| **video_generation** | OpenAI/SGLang text-to-video (`/v1/videos`) | ❌ | ✅ | Async polling; Sora / Wan2.1 / HunyuanVideo compatible; multipart-form requests |
| **image_retrieval** | Image search and retrieval endpoints | ✅ | ✅ | AIPerf serves NIM image retrieval / bounding-box detection at `/v1/infer` |
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
| **Endpoint Type** | `--endpoint-type` | ✅ | ✅ ⭐ | AIPerf supports 15+ endpoint types vs. GenAI-Perf's 12; see [detailed comparison](#endpoint-types-support-matrix) |
| **Server Metrics URL** | `--server-metrics-url` | ❌ | ✅ | AIPerf uses `--server-metrics` (enabled by default, auto-collects Prometheus metrics from the inference endpoint at `base_url + /metrics`). See the note below on GenAI-Perf's flag name. |
| **Streaming** | `--streaming` | ✅ | ✅ | |
| **URL** | `-u URL`<br/>`--url` | ✅ | ✅ | |
| **Request Timeout** | `--request-timeout-seconds` | ❌ | ✅ | |
| **API Key** | `--api-key` | ✅ | ✅ ⭐ | GenAI-Perf has no dedicated flag — users must pass `-H 'Authorization: Bearer ...'` manually |
| **Request Content Type** | `--request-content-type`<br/>`{application/json,multipart/form-data}` | ❌ | ✅ | Switch between JSON and multipart-form encoding (required by some video-gen servers) |

> [!NOTE]
> **GenAI-Perf's `--server-metrics-url` is misleadingly named.** Despite the "server metrics" label, the flag points GenAI-Perf at a **Triton / DCGM telemetry endpoint** (GPU power, utilization, memory) — it is *not* a general Prometheus inference-server metrics scraper. AIPerf splits this into two clearly-scoped flags:
> - `--server-metrics` — Prometheus inference-server metrics from the model endpoint (`base_url + /metrics`). Enabled by default; pass additional endpoint URLs to scrape extra targets.
> - `--gpu-telemetry` — GPU telemetry collection. Supports both the **DCGM exporter HTTP endpoint** (default; `localhost:9400` + `localhost:9401`) and the **local `pynvml` library** (pass `pynvml`). Custom DCGM exporter URLs and a `dashboard` realtime view are also accepted.
>
> If you're porting a GenAI-Perf invocation, `--server-metrics-url http://node:9400` maps to AIPerf's `--gpu-telemetry http://node:9400`, **not** to `--server-metrics`.

---

## **Input Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Extra Inputs** | `--extra-inputs` | ✅ | ✅ | |
| **Custom Headers** | `--header -H` | ✅ | ✅ | |
| **YAML Config File** | `-f --config` | ✅ (separate `config` subcommand) | ✅ | AIPerf passes YAML to `profile -f`; CLI flags override file values |
| **Input File** | `--input-file` | ✅ | ✅ | |
| **Inline Records in YAML** | `dataset.records:` (YAML only) | ❌ | ✅ | Embed dataset rows directly in the YAML config; >500 records emits a warning |
| **Dataset Entries** | `--num-dataset-entries --num-prompts` | ✅ | ✅ | GenAI-Perf and AIPerf both accept this flag. In AIPerf it is collapsed with `--num-sessions / --conversation-num / --num-conversations` into a single conversation count; GenAI-Perf keeps `--num-dataset-entries` and `--num-sessions` distinct (see [Session Configuration](#sessionconversation-configuration-multi-turn)). |
| **Public Dataset** | `--public-dataset` | ❌ | ✅ | sharegpt, aimo, mmstar, vision_arena, llava_onevision, speed_bench_* (50+ subsets), librispeech, voxpopuli, gigaspeech, ami, spgispeech, instruct_coder, blazedit_5k, blazedit_10k, ... |
| **HuggingFace Subset Override** | `--hf-subset` | ❌ | ✅ | Override the HF subset/config for HF-backed public datasets |
| **Custom Dataset Type** | `--custom-dataset-type`<br/>`{single_turn,multi_turn,random_pool,mooncake_trace,bailian_trace,burst_gpt_trace,sagemaker_data_capture}` | ❌ | ✅ | GenAI-Perf infers dataset type from input file format |
| **Dataset Sampling Strategy** | `--dataset-sampling-strategy`<br/>`{sequential,random,shuffle}` | ❌ | ✅ | Controls how entries are drawn during benchmarking |
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
| **Generate Plots** | `--generate-plots` | ✅ | ✅ ⭐ | AIPerf replaces inline plot generation with the dedicated `aiperf plot` subcommand: dashboard mode, Pareto overlays across runs, configurable plot envelope, auto-plot hook |
| **Auto-Plot After Profile** | `--auto-plot --no-auto-plot` | ❌ | ✅ | Auto-runs `aiperf plot` on the artifact dir after the benchmark completes; honored by recipe defaults |
| **Plot Required (Strict)** | `--plot-required` | ❌ | ✅ | Treat auto-plot failures as fatal (non-zero exit) |
| **Export Level** | `--export-level --profile-export-level`<br/>`{summary,records,raw}` | ❌ | ✅ | Controls whether per-record and raw request/response files are emitted alongside the summary |
| **Time-Sliced Metrics** | `--slice-duration` | ❌ | ✅ | Window the benchmark timeline into fixed slices and compute metrics per slice |
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
| **Audio Format** | `--audio-format`<br/>`{wav,mp3,random}` | 🟡 | ✅ | GenAI-Perf supports `{wav, mp3}` only; AIPerf adds `random` |
| **Audio Depths** | `--audio-depths` | ✅ | ✅ | |
| **Audio Sample Rates** | `--audio-sample-rates` | ✅ | ✅ | |
| **Audio Number of Channels** | `--audio-num-channels` | ✅ | ✅ | GenAI-Perf accepts `{1, 2}`; AIPerf accepts `{0, 1, 2}` (0 disables) |

---

## **Image Input Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Image Width Mean** | `--image-width-mean` | ✅ | ✅ | |
| **Image Width Stddev** | `--image-width-stddev` | ✅ | ✅ | |
| **Image Height Mean** | `--image-height-mean` | ✅ | ✅ | |
| **Image Height Stddev** | `--image-height-stddev` | ✅ | ✅ | |
| **Image Format** | `--image-format`<br/>`{png,jpeg,random}` | 🟡 | ✅ | GenAI-Perf supports `{png, jpeg}` only; AIPerf adds `random` |

---

## **Video Input Configuration**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Video Batch Size** | `--video-batch-size --batch-size-video` | ❌ | ✅ | Set to 0 to disable video inputs |
| **Video Duration** | `--video-duration` | ❌ | ✅ | Seconds per clip; requires FFmpeg |
| **Video FPS** | `--video-fps` | ❌ | ✅ | Frames per second |
| **Video Width/Height** | `--video-width --video-height` | ❌ | ✅ | Resolution in pixels (both or neither) |
| **Video Synth Type** | `--video-synth-type`<br/>`{moving_shapes,grid_clock,noise}` | ❌ | ✅ | Synthetic content generator |
| **Video Format** | `--video-format`<br/>`{webm,mp4}` | ❌ | ✅ | Container format |
| **Video Codec** | `--video-codec` | ❌ | ✅ | Any FFmpeg-supported codec (libvpx-vp9, libx264, h264_nvenc, ...) |
| **Embedded Audio Track** | `--video-audio-num-channels --video-audio-sample-rate --video-audio-codec --video-audio-depth` | ❌ | ✅ | Optional audio mux for video clips |
| **Download Video Content** | `--download-video-content` | ❌ | ✅ | Include video download time in request latency |

---

## **Multi-Run / Confidence Reporting**

> [!NOTE]
> AIPerf can repeat the same benchmark N times and report mean / std / confidence-interval / coefficient-of-variation across runs, optionally stopping early once a target metric stabilizes.

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Number of Profile Runs** | `--num-profile-runs` | ❌ | ✅ | 1-10 runs; >1 enables aggregate statistics |
| **Profile Run Cooldown** | `--profile-run-cooldown-seconds` | ❌ | ✅ | Stabilization gap between runs |
| **Confidence Level** | `--confidence-level` | ❌ | ✅ | 0.90 / 0.95 (default) / 0.99 CI width |
| **Disable Warmup After First** | `--profile-run-disable-warmup-after-first` | ❌ | ✅ | First run warms, rest measure steady state |
| **Consistent Seed Across Runs** | `--set-consistent-seed` | ❌ | ✅ | Auto-pin `--random-seed=42` for valid statistics |
| **Vary Seed Per Trial** | `--vary-seed-per-trial` | ❌ | ✅ | Capture input-noise + runtime-noise variance |
| **Adaptive Convergence Stopping** | `--convergence-metric --convergence-stat --convergence-threshold --convergence-mode` | ❌ | ✅ | Stop early when CI width, CV, or KS-distribution stabilizes |

---

## **Parameter Sweeping**

> [!NOTE]
> AIPerf folds GenAI-Perf's `analyze` subcommand into `profile` via three composable mechanisms: magic-list CLI flags, `--variant` scenarios, and YAML `sweep:` blocks. Multi-cell sweeps stream a per-cell results table; QMC sweeps additionally write a `sampling_design.json` for reproducibility.

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Magic-List CLI Flags** | `--concurrency 1,10,100` `--request-rate 50,100,200` `--isl 128,512,2048` `--osl ...` `--isl-stddev ...` `--osl-stddev ...` `--conversation-turn-mean ...` | 🟡 | ✅ | Any CLI flag in the allowlist accepts a comma list and triggers a sweep |
| **Grid Sweep** | `--sweep-type grid` (default) | ✅ (via `analyze`) | ✅ | Cartesian product of all magic-list flags |
| **Zip Sweep** | `--sweep-type zip` | ❌ | ✅ | Lockstep element-wise pairing; YAML form: `sweep: {type: zip}` |
| **Scenario Sweep** | `--variant --sweep-variant` | ❌ | ✅ | Repeatable `[name:] key=value, ...` per occurrence; emits a `ScenarioSweep`. GenAI-Perf's `analyze` sweeps one stimulus at a time (`{batch_size, concurrency, num_dataset_entries, input_sequence_length, request_rate}`), so multi-parameter scenarios are not expressible. |
| **Quasi-Monte-Carlo (Sobol)** | YAML `sweep: {type: sobol}` | ❌ | ✅ | Low-discrepancy quasi-random sampling over continuous + integer dimensions |
| **Latin Hypercube Sampling** | YAML `sweep: {type: latin_hypercube}` | ❌ | ✅ | Stratified sampling alternative to Sobol |
| **Sweep Variation Cooldown** | `--parameter-sweep-cooldown-seconds` | ❌ | ✅ | Inter-variation pause |
| **Same Seed Across Variations** | `--parameter-sweep-same-seed` | ❌ | ✅ | Correlated comparisons vs. independent draws |
| **Sweep Order** | `--parameter-sweep-mode`<br/>`{repeated,independent}` | ❌ | ✅ | Outer loop = trials or variations |
| **Live Sweep Table** | (auto) / `--no-sweep-table` | ❌ | ✅ | Per-cell streaming results table; auto-suppressed for non-TTY, dashboard UI, or single-cell sweeps |
| **Sampling Design Artifact** | `sweep_aggregate/sampling_design.json` | ❌ | ✅ | Emitted only for Sobol / Latin Hypercube (QMC) sweeps |
| **YAML Config Driven** | `dataset:` `phases:` `sweep:` `multi_run:` blocks | ✅ (separate `config` cmd) | ✅ | Single AIPerf YAML drives sweep + multi-run + plot envelope; `aiperf config expand` previews variations without running |

---

## **Adaptive Search / Bayesian Optimization**

> [!NOTE]
> AIPerf ships a native Bayesian-optimization search planner (Optuna + BoTorch preset, Hvarfner-DSP Matern-5/2 kernel), with native multi-objective Pareto support (qLogNEHVI), outcome constraints, posterior-regret stopping, and a curated set of preset "search recipes". GenAI-Perf does not offer adaptive search.

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Search Space** | `--search-space 'path:lo,hi[:kind]'` | ❌ | ✅ | Repeatable; CLI grammar supports `int` and `real` dimensions. YAML `sweep.search_space[]` also supports `prior: log-uniform`. |
| **Search Metric** | `--search-metric` | ❌ | ✅ | Tag from `RunResult.summary_metrics` |
| **Search Stat** | `--search-stat`<br/>`{avg,p50,p90,p95,p99}` | ❌ | ✅ | |
| **Search Direction** | `--search-direction` | ❌ | ✅ | Maximize / minimize |
| **Search Iterations** | `--search-max-iterations --search-initial-points --search-random-seed` | ❌ | ✅ | Sobol seed phase + GP fit + stopping |
| **Search Planner Plugin** | `--search-planner`<br/>`{bayesian,monotonic_sla,smooth_isotonic,optuna}` | ❌ | ✅ | `bayesian` is curated Optuna+BoTorch preset; third-party planners registerable |
| **Optuna Sampler** | `--optuna-sampler`<br/>`{tpe,gp,botorch}` | ❌ | ✅ | `--search-planner=optuna` expert mode |
| **Optuna Acquisition** | `--optuna-acquisition`<br/>`{logei,qlogei,qnei,qlognei,qehvi,qnehvi,qlognehvi}` | ❌ | ✅ | Modern noisy-EI defaults; multi-objective variants gated on `len(objectives) > 1` |
| **Posterior-Regret Stopping** | `--optuna-terminator`<br/>`{regret,emmr,none}` | ❌ | ✅ | RegretBoundEvaluator (Makarova 2022) / EMMR (Ishibashi 2023) |
| **Percentile Pooling** | `--search-percentile-pooling`<br/>`{mean,pooled}` | ❌ | ✅ | Pool raw samples across trials for tail-correct percentile objectives |
| **SLA Filter** | `--search-sla 'metric:stat:op:threshold'` | ❌ | ✅ | Repeatable; outcome-constraint or hard filter |
| **Multi-Objective Pareto** | `objectives: [...]` (YAML) | ❌ | ✅ | qNEHVI / qLogNEHVI; emits Pareto front in `search_history.json` |
| **Pareto Overlay Rendering** | `aiperf plot <dir1> <dir2> ...` | ❌ | ✅ | Multi-directory invocation triggers the Pareto overlay handler |

---

## **Search Recipes (Preset Experiments)**

| Feature | CLI Option | GenAI-Perf | AIPerf | Notes |
|---------|------------|------------|---------|-------|
| **Named Recipe** | `--search-recipe` | ❌ | ✅ | Expands to a search-space + SLA filter + post-process pipeline |
| **Pareto Sweep** | `--search-recipe pareto-sweep --isl-osl-pairs '128/128,512/256,...'` | ❌ | ✅ | Multi-shape throughput/latency Pareto with paired ISL/OSL workloads |
| **Max Throughput under TTFT SLA** | `--search-recipe max-throughput-ttft-sla --ttft-sla-ms` | ❌ | ✅ | Log-uniform concurrency prior with TTFT constraint |
| **Max Throughput under ITL SLA** | `--search-recipe max-throughput-itl-sla --itl-sla-ms` | ❌ | ✅ | Streaming required |
| **Max Concurrency under SLA** | `--search-recipe max-concurrency-under-sla --tpot-sla-ms --e2e-sla-ms --error-rate-sla --search-style {smooth_isotonic,monotonic,bo,optuna,grid}` | ❌ | ✅ | Selectable 1D SLA-saturation strategy |
| **Max Goodput under SLO** | `--search-recipe max-goodput-under-slo --slo-attainment-fraction` | ❌ | ✅ | DistServe-style per-request SLO attainment (default 0.95) |
| **Concurrency Ramp / Degradation Knee** | `--search-recipe concurrency-ramp --degradation-threshold --degradation-metric-tag --degradation-stat --concurrency-min --concurrency-max --concurrency-steps` | ❌ | ✅ | Reports first concurrency where stat exceeds baseline × (1 + threshold) |
| **Prefill TTFT Curve** | `--search-recipe prefill-ttft-curve --isl-min --isl-max --isl-steps` | ❌ | ✅ | Log-spaced ISL ramp for prefill characterization |
| **Decode ITL Curve** | `--search-recipe decode-itl-curve --osl-min --osl-max --osl-steps` | ❌ | ✅ | Log-spaced OSL ramp for decode characterization |

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
| **Log Level** | `--log-level` | ❌ | ✅ | `{TRACE,DEBUG,INFO,NOTICE,WARNING,SUCCESS,ERROR,CRITICAL}` (Loguru; case-insensitive in practice) |
| **UI Type** | `--ui-type --ui`<br/>`{dashboard,simple,none}` | ❌ | ✅ | |
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
| **GPU Telemetry** | ✅ | ✅ ⭐ | AIPerf supports dual backends: DCGM exporter HTTP endpoints (default; `localhost:9400` + `localhost:9401`, custom URLs accepted) and local `pynvml`. GenAI-Perf is DCGM-only via `--server-metrics-url`. |
| **Streaming API support** | ✅ | ✅ | |
| **Multi-turn conversations** | ✅ | ✅ | Full multi-turn benchmarking with session tracking |
| **Payload scheduling** | ✅ | ✅ | Fixed schedule workloads |
| **Distributed testing** | ✅ | ✅ ⭐ | GenAI-Perf has post-hoc multi-node result aggregation via `process-export-files`. AIPerf runs a single federated benchmark across nodes via ZMQ-TCP service-to-service communication. |
| **Custom endpoints** | ✅ | ✅ ⭐ | AIPerf ships 15+ endpoint types incl. `responses`, `chat_embeddings`, `nim_embeddings`, `nim_rankings`, `cohere_rankings`, `image_generation`, `video_generation`, `image_retrieval`, `solido_rag`. See [Endpoint Types Support Matrix](#endpoint-types-support-matrix). |
| **Synthetic data generation** | ✅ | ✅ | |
| **Bring Your Own Data (BYOD)** | ✅ | ✅ | Custom dataset support |
| **Audio input support** | ✅ | ✅ | Both tools synthesize audio inputs (WAV/MP3). Neither computes audio-specific metrics (e.g. WER, audio-token-rate) |
| **Vision metrics** | ✅ | ✅ | Image-specific performance metrics |
| **Image generation benchmarking** | ❌ | ✅ | Text-to-image with raw export for image extraction |
| **Video input benchmarking** | ❌ | ✅ | Synthetic video generation (FFmpeg) for VLM endpoints with configurable codec, resolution, FPS, audio |
| **Live Metrics** | ❌ | ✅ | Live metrics display |
| **Dashboard UI** | ❌ | ✅ | Dashboard UI |
| **Reasoning token parsing** | ❌ | ✅ | Parsing of reasoning tokens |
| **Arrival pattern control** | ❌ | ✅ | Constant, Poisson, Gamma distributions with tunable burstiness |
| **Prefill concurrency limiting** | ❌ | ✅ | Fine-grained prefill queueing control for TTFT behavior |
| **Gradual ramping** | ❌ | ✅ | Smooth ramp-up for concurrency and rate |
| **Duration-based benchmarking** | ❌ | ✅ | Time-based stop conditions with grace periods |
| **User-centric timing** | ❌ | ✅ | Per-user rate limiting for KV cache benchmarking |
| **Configurable warmup phase** | ✅ | ✅ ⭐ | GenAI-Perf has only `--warmup-request-count`. AIPerf adds duration-based, session-based, and full per-phase overrides (rate, concurrency, prefill, arrival pattern, grace period, ramping). |
| **HTTP trace metrics** | ❌ | ✅ | Detailed HTTP lifecycle timing (DNS, TCP, TLS, TTFB) |
| **Request cancellation** | ❌ | ✅ | Test timeout behavior and service resilience |
| **Timeslice metrics** | ❌ | ✅ | Per-timeslice metric breakdown |
| **Interactive plot dashboard** | ❌ | ✅ | Web-based exploration with dynamic metric selection and filtering |
| **Multi-run comparison plots** | ❌ | ✅ | Auto-detected Pareto curves and throughput analysis |
| **YAML-first configuration** | ✅ | ✅ ⭐ | GenAI-Perf has a separate `config` subcommand to run a YAML file. AIPerf threads a single YAML through dataset, phases, sweep, multi-run, and plot envelope with a `config init / expand / validate` lifecycle. |
| **Inline datasets in YAML** | ❌ | ✅ | Embed dataset records directly in the config file (no sidecar `.jsonl`) |
| **Plot config envelope** | ❌ | ✅ | YAML carries the visualization spec; auto-plot materializes `.aiperf-plot-config.yaml` for reproducibility |
| **Parameter sweeps** | ✅ (`analyze` subcommand) | ✅ ⭐ | GenAI-Perf's `analyze` sweeps one stimulus at a time. AIPerf folds sweeps into `profile` with magic-list CLI flags, multi-key scenario `--variant`, and YAML `sweep:` blocks (grid / zip / Sobol / Latin Hypercube). |
| **Confidence reporting** | ❌ | ✅ | Multi-run aggregation with CI / CV / KS-distribution convergence stopping |
| **Bayesian optimization** | ❌ | ✅ | Native Optuna+BoTorch search planner with Hvarfner-DSP kernel |
| **Multi-objective Pareto search** | ❌ | ✅ | qLogNEHVI acquisition, hypervolume stopping, Pareto front in `search_history.json` |
| **Search recipes (preset experiments)** | ❌ | ✅ | pareto-sweep, max-throughput-under-SLA, max-concurrency-under-SLA, max-goodput-under-SLO, concurrency-ramp, prefill-ttft-curve, decode-itl-curve |
| **Live sweep table with Pareto marking** | ❌ | ✅ | Per-cell streaming sweep table inline-marks Pareto-dominant cells (★) for recipes that declare `pareto_axes` |

---
