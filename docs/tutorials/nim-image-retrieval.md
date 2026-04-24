<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile NIM Image Retrieval with AIPerf

AIPerf supports benchmarking NVIDIA NIM image retrieval endpoints that detect objects and layout elements (tables, charts, titles, etc.) in images and return bounding box coordinates.

This guide covers profiling [NIM for Object Detection](https://docs.nvidia.com/nim/ingestion/object-detection/latest/overview.html) models such as `nemoretriever-page-elements` and `nemoretriever-graphic-elements` using the `/v1/infer` API.

---

## Section 1. Deploy the NIM Server

### Prerequisites

- NVIDIA GPU (Ampere, Hopper, or Lovelace architecture)
- Docker with NVIDIA runtime
- An [NGC API key](https://org.ngc.nvidia.com/setup/api-keys)

### Authenticate with NGC

```bash
export NGC_API_KEY=<your_api_key>
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

### Start the NIM Container

Launch the NIM for Object Detection (page elements):

```bash
export NIM_MODEL_NAME=nvidia/nemoretriever-page-elements-v3
export CONTAINER_NAME=$(basename $NIM_MODEL_NAME)
export IMG_NAME="nvcr.io/nim/nvidia/$CONTAINER_NAME:1.7.0"
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"

docker run -it --rm --name=$CONTAINER_NAME \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  $IMG_NAME
```

Wait for the server to start, then verify it is ready:

```bash
curl -s http://localhost:8000/v1/health/ready
```

### Verify with a Test Request

```bash
IMAGE_URL="https://assets.ngc.nvidia.com/products/api-catalog/nemo-retriever/object-detection/page-elements-example-1.jpg"
BASE64_IMAGE=$(curl -s "$IMAGE_URL" | base64 -w 0)

curl -s -X POST http://localhost:8000/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "input": [{
      "type": "image_url",
      "url": "data:image/jpeg;base64,'"$BASE64_IMAGE"'"
    }]
  }' | jq
```

**Sample Response:**
```json
{
  "data": [
    {
      "index": 0,
      "bounding_boxes": {
        "title": [
          { "x_min": 0.12, "y_min": 0.02, "x_max": 0.88, "y_max": 0.06, "confidence": 0.97 }
        ],
        "table": [
          { "x_min": 0.05, "y_min": 0.15, "x_max": 0.95, "y_max": 0.85, "confidence": 0.94 }
        ]
      }
    }
  ],
  "usage": { "images_size_mb": 0.42 }
}
```

Bounding box coordinates are normalized (0-1 range) relative to the top-left corner of the image.

---

## Section 2. Profile with Custom Images

Create a JSONL input file with image paths or URLs:

```bash
cat <<EOF > inputs.jsonl
{"image": "https://assets.ngc.nvidia.com/products/api-catalog/nemo-retriever/object-detection/page-elements-example-1.jpg"}
{"image": "/path/to/local/document_page.png"}
{"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."}
EOF
```

Each line should contain an `image` field with either:
- A URL to a remote image
- A local file path (automatically encoded to base64)
- A base64 data URL (passed through as-is)

Run AIPerf against the image retrieval endpoint:

```bash
aiperf profile \
    --endpoint-type image_retrieval \
    --model nvidia/nemoretriever-page-elements-v3 \
    --url localhost:8000 \
    --input-file inputs.jsonl \
    --custom-dataset-type single_turn \
    --request-count 20 \
    --concurrency 4
```

**Sample Output:**
```
INFO     Starting AIPerf System
INFO     Tokenization is disabled for this endpoint, skipping tokenizer configuration
INFO     AIPerf System is PROFILING

Profiling: 20/20 |████████████████████████| 100% [00:05<00:00]

INFO     Benchmark completed successfully

                  NVIDIA AIPerf | Image Retrieval Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                           Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│             Request Latency (ms) │  85.23 │  62.10 │ 142.56 │ 138.90 │ 125.30 │  82.45 │  21.34 │
│  Image Throughput (images/sec)   │  11.73 │   7.01 │  16.10 │   7.20 │   8.05 │  12.13 │   2.45 │
│       Image Latency (ms/image)   │  85.23 │  62.10 │ 142.56 │ 138.90 │ 125.30 │  82.45 │  21.34 │
│  Request Throughput (requests/sec) │  11.73 │    N/A │    N/A │    N/A │    N/A │    N/A │    N/A │
└──────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

Since this endpoint does not produce tokens, no TTFT or ITL metrics are reported. The primary metrics are request latency, image throughput, and image latency.

---

## Section 3. Profile with Multiple Images per Request

You can send multiple images in a single request:

```bash
cat <<EOF > multi_image_inputs.jsonl
{"images": ["https://example.com/page1.png", "https://example.com/page2.png"]}
{"images": ["/path/to/chart1.jpg", "/path/to/chart2.jpg", "/path/to/chart3.jpg"]}
EOF
```

```bash
aiperf profile \
    --endpoint-type image_retrieval \
    --model nvidia/nemoretriever-page-elements-v3 \
    --url localhost:8000 \
    --input-file multi_image_inputs.jsonl \
    --custom-dataset-type single_turn \
    --request-count 10 \
    --concurrency 2
```

When sending multiple images per request, the image throughput metric reflects the total number of images processed per second across all requests.

---

## Section 4. Pass Extra Parameters

Use `--extra-inputs` to pass additional parameters to the NIM endpoint:

```bash
aiperf profile \
    --endpoint-type image_retrieval \
    --model nvidia/nemoretriever-page-elements-v3 \
    --url localhost:8000 \
    --input-file inputs.jsonl \
    --custom-dataset-type single_turn \
    --extra-inputs threshold:0.5 \
    --request-count 20 \
    --concurrency 4
```

Extra inputs are merged into the request payload alongside the image data.

---

## Section 5. Using Other NIM Object Detection Models

The `image_retrieval` endpoint works with any NIM that accepts the `/v1/infer` API format. You can swap models by changing the Docker image:

| Model | Container Image | Description |
|-------|----------------|-------------|
| Page Elements v3 | `nvcr.io/nim/nvidia/nemoretriever-page-elements-v3:1.7.0` | Detects tables, charts, titles, paragraphs, headers/footers |
| Graphic Elements v1 | `nvcr.io/nim/nvidia/nemoretriever-graphic-elements-v1:latest` | Detects components within charts (axes, legends, data points) |
| Table Structure v1 | `nvcr.io/nim/nvidia/nemoretriever-table-structure-v1:latest` | Identifies table cells, rows, and columns |

All models share the same `/v1/infer` request/response format, so the same AIPerf `--endpoint-type image_retrieval` command works for each.
