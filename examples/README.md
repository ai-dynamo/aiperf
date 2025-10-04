<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Examples

Example scripts and data demonstrating how to work with AIPerf outputs.

---

## Scripts

### `parse_profile_export.py`

**Purpose:** Load and display `profile_export.jsonl` files using AIPerf's native Pydantic models.

**What it does:**
- Parses JSONL records into `MetricRecordInfo` objects
- Displays example record structure with Rich formatting
- Lists all available metrics with types and units

**Usage:**
```bash
# Use example artifacts
python examples/parse_profile_export.py

# Use your own data
python examples/parse_profile_export.py artifacts/my-run/profile_export.jsonl

# Async mode
python examples/parse_profile_export.py artifacts/my-run/profile_export.jsonl --async
```

---

## Example Artifacts

The `artifacts/run1/` directory contains sample output files from an actual AIPerf run.

### `profile_export.jsonl`

**Format:** JSON Lines (one record per line)
**Purpose:** Per-request metric data with full detail
**Size:** ~100 records from a sample benchmark run

Each line contains:
- Request metadata (worker ID, timestamps, conversation ID)
- Metrics for that specific request (latency, tokens, etc.)
- Error information (if request failed)

**When to use:** Analyzing individual request behavior, finding outliers, debugging issues.

### `profile_export_aiperf.json`

**Format:** Single JSON object
**Purpose:** Aggregated statistics across all requests
**Contains:** Min, max, mean, percentiles (p50, p95, p99, etc.) for each metric

**When to use:** Getting overall performance summary, comparing benchmarks.

### `profile_export_aiperf.csv`

**Format:** CSV
**Purpose:** Same aggregated statistics in spreadsheet-friendly format

**When to use:** Importing into Excel

---

## Data Models

AIPerf uses Pydantic models for type-safe data handling:

```python
from aiperf.common.models import (
    MetricRecordInfo,
    MetricRecordMetadata,
    MetricValue,
    ErrorDetails,
)
```

**Model Definitions:**
- [`MetricRecordInfo`](../aiperf/common/models/record_models.py#L97) - Complete per-request record structure
- [`MetricRecordMetadata`](../aiperf/common/models/record_models.py#L66) - Request metadata (timestamps, IDs, worker info)
- [`MetricValue`](../aiperf/common/models/record_models.py#L59) - Individual metric value with unit
- [`ErrorDetails`](../aiperf/common/models/error_models.py#L11) - Error information (code, type, message)

**Example: Successful Request**
```json
{
  "metadata": {
    "x_request_id": "dc9ffe9b-0f48-4863-9db7-e106ea64d094",
    "x_correlation_id": "6f24617d-f29b-4399-af87-b98a61fe43f7",
    "conversation_id": "af0041e4-aec8-412b-82a5-e6e2455b892a",
    "turn_index": 0,
    "timestamp_ns": 1759522419158355424,
    "worker_id": "worker_aec1f387",
    "record_processor_id": "record_processor_5cdb3e92",
    "credit_phase": "profiling"
  },
  "metrics": {
    "input_sequence_length": {"value": 550, "unit": "tokens"},
    "ttft": {"value": 3340.543979, "unit": "ms"},
    "request_count": {"value": 1, "unit": "requests"},
    "request_latency": {"value": 8214.202336, "unit": "ms"},
    "min_request_timestamp": {"value": 1759522419158355424, "unit": "ns"},
    "output_token_count": {"value": 62, "unit": "tokens"},
    "reasoning_token_count": {"value": 131, "unit": "tokens"},
    "ttst": {"value": 139.01211999999998, "unit": "ms"},
    "inter_chunk_latency": {
      "value": [139.01211999999998,137.504221,138.46932999999999,45.724841,22.820729,22.354865999999998,24.233856,21.191751,21.803236,21.112617,22.138389999999998,22.290063,21.568081,21.639267999999998,21.043643,21.699054999999998,21.465737,21.357903,23.664227,19.843211,21.429326,20.807807999999998,22.244322999999998,22.980819999999998,21.714579999999998,20.311258,22.320152,20.716417,21.489044,23.120455999999997,21.194105,21.747794,21.775451,22.772171,21.177619999999997,21.435968,22.72408,22.319703999999998,23.697609999999997,22.692925,24.573838,24.935859999999998,26.220257,31.696505,27.352555,24.474261,30.586574,22.706429,27.079940999999998,21.097013,21.312921,19.886108,21.975094,25.711636,23.944499999999998,22.047128,19.041073,25.347305,21.117617,20.374716,21.078395999999998,22.556409,21.256626,22.730458,20.697526999999997,24.304420999999998,19.036089999999998,22.208375999999998,21.108458,22.866515,26.124654,19.439919,24.660149,24.480218999999998,22.055654999999998,24.99418,18.583989,23.828048,22.653662999999998,20.263586,22.421452,22.796861,23.564021,22.431328,22.228718999999998,21.330883,21.859503,22.474016,22.873683,22.787454999999998,22.573733999999998,21.460922,22.424144,22.442114999999998,23.179195,22.802578999999998,22.545786,22.882702,23.31232,23.126859,21.893006,23.557437999999998,22.776183,22.061291999999998,22.107775999999998,22.255364,22.322226999999998,24.980131999999998,21.467501,21.797259999999998,23.437003999999998,23.993665999999997,22.305018999999998,23.036853999999998,22.524950999999998,22.406306,22.918474,22.922335999999998,21.904897,21.565794,23.226157999999998,23.259197999999998,23.434093999999998,21.758516999999998,22.842456,22.888417999999998,21.407372,22.814517,22.408683,22.539944,160.85134,23.339579999999998,22.765987,22.429622,22.025340999999997,22.615395,22.957057,23.911932,22.003268,22.03979,23.155224,22.854999,23.844901,23.013745,22.209705,23.692,22.305362,22.788823,24.418011999999997,21.410004,23.309737,22.293789,24.580631999999998,21.682264,22.708857,22.872097,23.393947,23.339647,23.22015,24.162468999999998,22.352579,24.407208999999998,23.268773,23.927725,23.887949,22.625245,23.777784999999998,23.140172999999997,22.655293,23.344238999999998,23.776709999999998,22.741847,24.011459,22.901256,24.119477999999997,21.972716,23.987218,22.558432999999997,22.693851,22.350789,23.023360999999998,22.424141,22.478153,26.871579999999998,23.642765999999998,19.764049999999997,23.363139,22.169117999999997,23.956905,21.800013999999997,22.825948999999998,24.294238],
      "unit": "ms"
    },
    "output_sequence_length": {"value": 193, "unit": "tokens"},
    "max_response_timestamp": {"value": 1759522427372557760, "unit": "ns"},
    "inter_token_latency": {"value": 25.383637276041668, "unit": "ms"},
    "output_token_throughput_per_user": {"value": 39.395457361969534, "unit": "tokens/sec/user"}
  },
  "error": null
}
```

**Example: Failed Request**
```json
{
  "metadata": {
    "x_request_id": "c087da26-5552-4785-958a-cad75c7b22de",
    "x_correlation_id": "51a3af05-bcd3-4ad6-ab75-3eec065ffe6c",
    "conversation_id": "82c50e66-608b-4cc1-bcab-a71090077120",
    "turn_index": 0,
    "timestamp_ns": 1759522419154829652,
    "worker_id": "worker_1d62deaa",
    "record_processor_id": "record_processor_ef1716ef",
    "credit_phase": "profiling"
  },
  "metrics": {
    "error_request_count": {"value": 1, "unit": "requests"}
  },
  "error": {
    "code": 499,
    "type": "RequestCancellationError",
    "message": "Request was cancelled after 0.000 seconds"
  }
}
```

---
