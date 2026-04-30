<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay SageMaker Data Capture Traces

AIPerf supports replaying production traffic captured by [Amazon SageMaker Data Capture](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-capture-endpoint.html). This enables benchmarking inference servers using real request patterns and prompts recorded from SageMaker real-time endpoints.

The loader sends the exact captured prompts (literal replay via the `messages` array) with original request timing, enabling accurate A/B comparisons when migrating models, changing instance types, or upgrading serving frameworks.

---

## Prerequisites

- A SageMaker real-time endpoint with Data Capture enabled (captures both input and output)
- Captured data synced from S3 to local disk
- The captured endpoint must use the OpenAI-compatible chat completions API (`messages` array in the request payload)

---

## SageMaker Data Capture Format

Data Capture writes JSONL files to S3, partitioned by hour:

```text
s3://<bucket>/<prefix>/<endpoint-name>/<variant-name>/yyyy/mm/dd/hh/<uuid>.jsonl
```

Each JSONL line contains the full request and response payloads with timing metadata:

```json
{
  "captureData": {
    "endpointInput": {
      "observedContentType": "application/json",
      "mode": "INPUT",
      "data": "{\"messages\":[{\"role\":\"user\",\"content\":\"What is AI?\"}],\"max_tokens\":50}",
      "encoding": "JSON"
    },
    "endpointOutput": {
      "observedContentType": "application/json",
      "mode": "OUTPUT",
      "data": "{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":30,\"total_tokens\":42},...}",
      "encoding": "JSON"
    }
  },
  "eventMetadata": {
    "eventId": "e4378ff2-2b43-4031-a21f-401bb3c3e038",
    "inferenceTime": "2026-04-29T00:03:18Z"
  },
  "eventVersion": "0"
}
```

---

## Download and Replay

Sync captured data from S3 and point AIPerf at the directory:

```bash
# Sync all capture files (preserves hourly directory structure)
aws s3 sync \
  s3://my-bucket/datacapture/my-endpoint/primary/ \
  ./captured_data/

# Replay against a target server
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --url localhost:8000 \
    --input-file ./captured_data/ \
    --custom-dataset-type sagemaker_data_capture \
    --fixed-schedule \
    --fixed-schedule-auto-offset
```

The loader recursively finds all `.jsonl` files in the directory, parses them, and sorts records by timestamp. No manual file concatenation is needed.

Single-file input also works:

```bash
# Concatenate if preferred
find captured_data/ -name "*.jsonl" -exec cat {} + > all_captures.jsonl

aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --url localhost:8000 \
    --input-file all_captures.jsonl \
    --custom-dataset-type sagemaker_data_capture \
    --fixed-schedule \
    --fixed-schedule-auto-offset
```

---

## Replay a Time Window

Use timestamp offsets to replay a subset of the captured traffic:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --url localhost:8000 \
    --input-file ./captured_data/ \
    --custom-dataset-type sagemaker_data_capture \
    --fixed-schedule \
    --fixed-schedule-auto-offset \
    --fixed-schedule-end-offset 300000
```

This replays only the first 5 minutes (300,000 ms) of captured traffic.

---

## Enabling Data Capture on Your Endpoint

When creating the endpoint configuration, include `DataCaptureConfig` with `JsonContentTypes` to store payloads as raw JSON (not base64):

```python
import boto3

client = boto3.client("sagemaker")

client.create_endpoint_config(
    EndpointConfigName="my-endpoint-config-with-capture",
    ProductionVariants=[{
        "VariantName": "primary",
        "ModelName": "my-model",
        "InitialInstanceCount": 1,
        "InstanceType": "ml.g5.xlarge",
        "InitialVariantWeight": 1.0,
    }],
    DataCaptureConfig={
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,
        "DestinationS3Uri": "s3://my-bucket/datacapture",
        "CaptureOptions": [
            {"CaptureMode": "Input"},
            {"CaptureMode": "Output"},
        ],
        "CaptureContentTypeHeader": {
            "JsonContentTypes": ["application/json"],
        },
    },
)
```

> [!NOTE]
> Setting `JsonContentTypes` ensures payloads are stored as raw JSON. Without it, SageMaker base64-encodes the data by default. The AIPerf loader handles both encodings.

---

## Known Limitations

- **Second-level timestamp precision**: `inferenceTime` has no fractional seconds. At high QPS, requests sharing the same second fire in rapid succession.
- **No streaming capture**: `InvokeEndpointWithResponseStream` responses are not captured by SageMaker. Output token counts may be missing for streaming endpoints.
- **Single-turn only**: Each captured record is an independent request. No multi-turn session linking.
- **OpenAI-compatible only**: The captured payload must contain a `messages` array. Non-chat endpoints are not supported.

---

## Related Tutorials

- [Trace Replay with Mooncake Traces](../benchmark-modes/trace-replay.md) - Mooncake FAST'25 trace replay
- [Bailian Traces](bailian-trace.md) - Bailian production trace replay
- [Fixed Schedule](fixed-schedule.md) - Precise timestamp-based execution for any dataset
