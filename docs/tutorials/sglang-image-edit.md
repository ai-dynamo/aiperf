---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: SGLang Image Edit
---

# Profile Image Edit (Image-to-Image) Models with AIPerf

## Overview
This guide shows how to benchmark image-to-image (TI2I) APIs using a Docker-based server and AIPerf. You'll learn how to:

- Set up the server (FLUX.2-Klein-4B on SGLang)
- Run the benchmark with synthetic reference images or your own input file
- View the results and extract the edited images

The endpoint follows the OpenAI Image Edit shape: prompt + reference image are POSTed to `/v1/images/edits` as `multipart/form-data`. AIPerf auto-defaults `request_content_type` to multipart for `image_edit`, so you don't need to pass `--request-content-type` explicitly.

## References
For the most up-to-date information, please refer to the following resources:
- [OpenAI Image Edit API](https://platform.openai.com/docs/api-reference/images/createEdit)
- [SGLang Multimodal Gen — `/v1/images/edits` route](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py)
- [FLUX.2-Klein-4B on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

## Setting up the server

**Login to Hugging Face, and accept the terms of use for** [FLUX.2-Klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

**Export your Hugging Face token as an environment variable:**
```bash
export HF_TOKEN=<your-huggingface-token>
```

**Start the Docker container:**
```bash
docker run --gpus all \
    --shm-size 32g \
    -it \
    --rm \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:dev
```

> [!NOTE]
> The following steps are to be performed _inside_ the Docker container. `lmsysorg/sglang:dev` ships the diffusion stack ready to run — no extra `pip install` step is needed for FLUX.2-Klein-4B.

**Set the server arguments:**
> [!WARNING]
> These arguments set up FLUX.2-Klein-4B on a single GPU at port 30000.
> Adjust the model path, GPU count, or port to match your environment.
> The flags below come from upstream SGLang multimodal_gen and may change over time — treat the [SGLang Multimodal Gen CLI](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md) as the source of truth if any flag here is rejected.
```bash
SERVER_ARGS=( --model-path black-forest-labs/FLUX.2-klein-4B --num-gpus 1 --port 30000 --host 0.0.0.0 --warmup --enable-torch-compile )
```

**Start the server:**
```bash
sglang serve "${SERVER_ARGS[@]}"
```

**Wait until the server is ready** (watch the logs for the following message):
```bash
Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

## Running the benchmark (basic usage)

> [!NOTE]
> The following steps are to be performed on your local machine (_outside_ the Docker container).

### Image Edit Using Synthetic Reference Images
The simplest path: AIPerf generates a synthetic reference image for every request and pairs it with a synthetic prompt. The mock image bytes are uploaded as the multipart `image` field — the server processes the request end-to-end just like a real one.

```bash
aiperf profile \
  --model black-forest-labs/FLUX.2-klein-4B \
  --tokenizer gpt2 \
  --url http://localhost:30000 \
  --endpoint-type image_edit \
  --image-batch-size 1 \
  --image-width-mean 512 \
  --image-height-mean 512 \
  --extra-inputs size:512x512 \
  --extra-inputs num_inference_steps:4 \
  --extra-inputs guidance_scale:1.0 \
  --warmup-request-count 5 \
  --request-count 50 \
  --concurrency 2
```

**Done!** This sends 50 requests to `http://localhost:30000/v1/images/edits` with multipart-encoded prompt + reference image, plus diffusion-specific extras (`size`, `num_inference_steps`, `guidance_scale`).

**Sample Output (shape only — exact numbers will depend on your hardware):**
```
                                  NVIDIA AIPerf | Image Edit Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓
┃                            Metric ┃  avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p50 ┃ std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩
│              Request Latency (ms) │  ... │ ... │ ... │ ... │ ... │ ... │ ... │
│    Input Sequence Length (tokens) │  ... │ ... │ ... │ ... │ ... │ ... │ ... │
│     Image Throughput (images/sec) │  ... │ ... │ ... │ ... │ ... │ ... │ ... │
│          Image Latency (ms/image) │  ... │ ... │ ... │ ... │ ... │ ... │ ... │
│ Request Throughput (requests/sec) │  ... │ N/A │ N/A │ N/A │ N/A │ N/A │ N/A │
│          Request Count (requests) │  ... │ N/A │ N/A │ N/A │ N/A │ N/A │ N/A │
└───────────────────────────────────┴──────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

### Image Edit Using an Input File
For deterministic prompt + reference image sequences, use a JSONL input file. Each line must include both the prompt (`text`) and the reference image (`image`, a local path or URL) — the `image_edit` endpoint rejects turns without a reference image, and the `single_turn` loader does not synthesize one.

**Create an input file** (replace the paths/URLs with real reference images you want to edit):
```bash
cat > edit_prompts.jsonl << 'EOF'
{"text": "Convert this scene to a watercolor painting", "image": "/path/to/ref1.png"}
{"text": "Make the background a sunset", "image": "/path/to/ref2.png"}
{"text": "Add snow to the trees", "image": "https://example.com/ref3.png"}
EOF
```

**Run the benchmark:**
```bash
aiperf profile \
  --model black-forest-labs/FLUX.2-klein-4B \
  --tokenizer gpt2 \
  --url http://localhost:30000 \
  --endpoint-type image_edit \
  --input-file edit_prompts.jsonl \
  --custom-dataset-type single_turn \
  --extra-inputs size:512x512 \
  --extra-inputs num_inference_steps:4 \
  --concurrency 1 \
  --request-count 3
```

## Understanding the Metrics

Image edit shares its metric set with image generation; both endpoints report image-level throughput/latency on top of the standard request-level metrics. There are no token-streaming metrics (TTFT, ITL) because the edited image is returned as a single response.

| Metric | Description |
|---|---|
| **Request Latency (ms)** | End-to-end time per request — from sending the multipart body to receiving the edited image. |
| **Input Sequence Length (tokens)** | Token count of the prompt portion only; the reference image is uploaded separately as binary and does not contribute. |
| **Image Throughput (images/sec)** | Number of edited images returned per second across all concurrent workers. |
| **Image Latency (ms/image)** | Per-image latency; equals request latency when each request returns one image. |
| **Request Throughput (requests/sec)** | Sustained request rate. |
| **Request Count (requests)** | Total completed requests. |

> [!TIP]
> The first request typically pays a `torch.compile` cold-start cost (multiple seconds). Use `--warmup-request-count` to exclude warmup requests from the reported metrics.

## Running the benchmark (advanced usage)

Use `--export-level raw` to capture the raw input/output payloads, which lets you extract the edited images afterwards.

```bash
aiperf profile \
  --model black-forest-labs/FLUX.2-klein-4B \
  --tokenizer gpt2 \
  --url http://localhost:30000 \
  --endpoint-type image_edit \
  --input-file edit_prompts.jsonl \
  --custom-dataset-type single_turn \
  --extra-inputs size:512x512 \
  --extra-inputs num_inference_steps:4 \
  --concurrency 1 \
  --request-count 3 \
  --export-level raw
```

### Viewing the edited images

The edited images come back as base64 strings inside each response. You can reuse the same extraction script from the [Image Generation tutorial](image-generation.md#viewing-the-generated-images) — the response shape is identical. Point it at the `image_edit` artifacts directory:

```bash
python extract_images.py \
  artifacts/black-forest-labs_FLUX.2-klein-4B-openai-image_edit-concurrency1/profile_export_raw.jsonl \
  extracted_edits
```

## Conclusion

You've set up an image-to-image diffusion server, benchmarked it with both synthetic and file-driven prompts, and seen the metric set AIPerf reports for `image_edit`. From here you can sweep over `num_inference_steps`, `guidance_scale`, resolution, or concurrency to map the perf trade-offs of your model and hardware.
