<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Working with Profile Export Files

This guide demonstrates how to programmatically work with AIPerf benchmark output files using the native Pydantic data models.

## Overview

AIPerf generates multiple output formats after each benchmark run, each optimized for different analysis workflows:

- [**`inputs.json`**](#input-dataset-json) - Complete input dataset with formatted payloads for each request
- [**`profile_export.jsonl`**](#per-request-records-jsonl) - Per-request metric records in JSON Lines format with one record per line
- [**`profile_export_aiperf.json`**](#aggregated-statistics-json) - Aggregated statistics and user configuration as a single JSON object
- [**`profile_export_aiperf.csv`**](#aggregated-statistics-csv) - Aggregated statistics in CSV format

## Data Models

AIPerf uses Pydantic models for type-safe parsing and validation of all benchmark output files. These models ensure data integrity and provide IDE autocompletion support.

### Core Models

```python
from aiperf.common.models import (
    MetricRecordInfo,
    MetricRecordMetadata,
    MetricValue,
    ErrorDetails,
    InputsFile,
    SessionPayloads,
)
```

| Model | Description | Source |
|-------|-------------|--------|
| `MetricRecordInfo` | Complete per-request record including metadata, metrics, and error information | [record_models.py](../../aiperf/common/models/record_models.py) |
| `MetricRecordMetadata` | Request metadata: timestamps, IDs, worker identifiers, and phase information | [record_models.py](../../aiperf/common/models/record_models.py) |
| `MetricValue` | Individual metric value with associated unit of measurement | [record_models.py](../../aiperf/common/models/record_models.py) |
| `ErrorDetails` | Error information including HTTP code, error type, and descriptive message | [error_models.py](../../aiperf/common/models/error_models.py) |
| `InputsFile` | Container for all input dataset sessions with formatted payloads for each turn | [dataset_models.py](../../aiperf/common/models/dataset_models.py) |
| `SessionPayloads` | Single conversation session with session ID and list of formatted request payloads | [dataset_models.py](../../aiperf/common/models/dataset_models.py) |

## Output File Formats

### Input Dataset (JSON)

**File:** `artifacts/my-run/inputs.json`

A structured representation of all input datasets converted to the payload format used by the endpoint.

**Structure:**
```json
{
  "data": [
    {
      "session_id": "a5cdb1fe-19a3-4ed0-9e54-ed5ed6dc5578",
      "payloads": [
        { ... }  // formatted payload based on the endpoint type.
      ]
    }
  ]
}
```

**Key fields:**
- `session_id`: Unique identifier for the conversation. This can be used to correlate inputs with results.
- `payloads`: Array of formatted request payloads (one per turn in multi-turn conversations)

### Per-Request Records (JSONL)

**File:** `artifacts/my-run/profile_export.jsonl`

The JSONL output contains one record per line, for each request sent during the benchmark. Each record includes request metadata, computed metrics, and error information if the request failed.

#### Successful Request Record

```json
{
  "metadata": {
    "session_num": 45,
    "x_request_id": "7609a2e7-aa53-4ab1-98f4-f35ecafefd25",
    "x_correlation_id": "32ee4f33-cfca-4cfc-988f-79b45408b909",
    "conversation_id": "77aa5b0e-b305-423f-88d5-c00da1892599",
    "turn_index": 0,
    "request_start_ns": 1759813207532900363,
    "request_ack_ns": 1759813207650730976,
    "request_end_ns": 1759813207838764604,
    "worker_id": "worker_359d423a",
    "record_processor_id": "record_processor_1fa47cd7",
    "benchmark_phase": "profiling",
    "was_cancelled": false,
    "cancellation_time_ns": null
  },
  "metrics": {
    "input_sequence_length": {"value": 550, "unit": "tokens"},
    "ttft": {"value": 255.88656799999998, "unit": "ms"},
    "request_latency": {"value": 297.52522799999997, "unit": "ms"},
    "output_token_count": {"value": 9, "unit": "tokens"},
    "ttst": {"value": 4.8984369999999995, "unit": "ms"},
    "inter_chunk_latency": {"value": [4.898437, 5.316006, 4.801489, 5.674918, 4.811467, 5.097998, 5.504797, 5.533548], "unit": "ms"},
    "output_sequence_length": {"value": 9, "unit": "tokens"},
    "inter_token_latency": {"value": 5.2048325, "unit": "ms"},
    "output_token_throughput_per_user": {"value": 192.1291415237666, "unit": "tokens/sec/user"}
  },
  "error": null
}
```

**Metadata Fields:**
- `session_num`: Sequential request number across the entire benchmark (0-indexed).
  - For single-turn conversations, this will be the request index across all requests in the benchmark.
  - For multi-turn conversations, this will be the index of the user session across all sessions in the benchmark.
- `x_request_id`: Unique identifier for this specific request. This is sent to the endpoint as the X-Request-ID header.
- `x_correlation_id`: Unique identifier for the user session. This is the same for all requests in the same user session for multi-turn conversations. This is sent to the endpoint as the X-Correlation-ID header.
- `conversation_id`: ID of the input dataset conversation. This can be used to correlate inputs with results.
- `turn_index`: Position within a multi-turn conversation (0-indexed), or 0 for single-turn conversations.
- `request_start_ns`: Epoch time in nanoseconds when request was initiated by AIPerf.
- `request_ack_ns`: Epoch time in nanoseconds when server acknowledged the request. This is only applicable to streaming requests.
- `request_end_ns`: Epoch time in nanoseconds when the last response was received from the endpoint.
- `worker_id`: ID of the AIPerf worker that executed the request against the endpoint.
- `record_processor_id`: ID of the AIPerf record processor that processed the results from the server.
- `benchmark_phase`: Phase of the benchmark. Currently only `profiling` is supported.
- `was_cancelled`: Whether the request was cancelled during execution (such as when `--request-cancellation-rate` is enabled).
- `cancellation_time_ns`: Epoch time in nanoseconds when the request was cancelled (if applicable).

**Metrics:**
See the [Complete Metrics Reference](../metrics_reference.md) page for a list of all metrics and their descriptions. Will always be null for failed requests.

#### Failed Request Record

```json
{
  "metadata": {
    "session_num": 80,
    "x_request_id": "c35e4b1b-6775-4750-b875-94cd68e5ec15",
    "x_correlation_id": "77ecf78d-b848-4efc-9579-cd695c6e89c4",
    "conversation_id": "9526b41d-5dbc-41a5-a353-99ae06a53bc5",
    "turn_index": 0,
    "request_start_ns": 1759879161119147826,
    "request_ack_ns": null,
    "request_end_ns": 1759879161119772754,
    "worker_id": "worker_6006099d",
    "record_processor_id": "record_processor_fdeeec8f",
    "benchmark_phase": "profiling",
    "was_cancelled": true,
    "cancellation_time_ns": 1759879161119772754
  },
  "metrics": {
    "error_isl": {"value": 550, "unit": "tokens"}
  },
  "error": {
    "code": 499,
    "type": "RequestCancellationError",
    "message": "Request was cancelled after 0.000 seconds"
  }
}
```

**Error Fields:**
- `code`: HTTP status code or custom error code
- `type`: Classification of the error (e.g., timeout, cancellation, server error). Typically the python exception class name.
- `message`: Human-readable error description


### Aggregated Statistics (JSON)

**File:** `artifacts/my-run/profile_export_aiperf.json`

A single JSON object containing statistical summaries (min, max, mean, percentiles) for all metrics across the entire benchmark run, as well as the user configuration used for the benchmark.

### Aggregated Statistics (CSV)

**File:** `artifacts/my-run/profile_export_aiperf.csv`

Contains the same aggregated statistics as the JSON format, but in a spreadsheet-friendly structure with one metric per row.

## Working with Output Data

AIPerf output files can be parsed using the native Pydantic models for type-safe data handling and analysis.

### Synchronous Loading
```python
from aiperf.common.models import MetricRecordInfo

def load_records(file_path: Path) -> list[MetricRecordInfo]:
    """Load artifacts/my-run/profile_export.jsonl file into structured Pydantic models in sync mode."""
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = MetricRecordInfo.model_validate_json(line)
                records.append(record)
    return records
```

### Asynchronous Loading

For large benchmark runs with thousands of requests, use async file I/O for better performance:

```python
import aiofiles
from aiperf.common.models import MetricRecordInfo

async def process_streaming_records_async(file_path: Path) -> None:
    """Load artifacts/my-run/profile_export.jsonl file into structured Pydantic models in async mode and process the streaming records."""
    async with aiofiles.open(file_path, encoding="utf-8") as f:
        async for line in f:
            if line.strip():
                record = MetricRecordInfo.model_validate_json(line)
                # ... Process the streaming records here ...
```

### Working with Input Datasets

Load and analyze the `inputs.json` file to understand what data was sent during the benchmark:

```python
import json
from pathlib import Path
from aiperf.common.models import InputsFile

def load_inputs_file(file_path: Path) -> InputsFile:
    """Load inputs.json file into structured Pydantic model."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return InputsFile.model_validate(data)

# Load inputs
inputs = load_inputs_file(Path("artifacts/my-run/inputs.json"))

# Analyze input characteristics
total_sessions = len(inputs.data)
total_turns = sum(len(session.payloads) for session in inputs.data)
multi_turn_sessions = sum(1 for session in inputs.data if len(session.payloads) > 1)

print(f"Total sessions: {total_sessions}")
print(f"Total turns: {total_turns}")
print(f"Multi-turn conversations: {multi_turn_sessions}")

# Extract prompt lengths for chat endpoints
for session in inputs.data:
    for turn_idx, payload in enumerate(session.payloads):
        if "messages" in payload:
            for message in payload["messages"]:
                content_length = len(message["content"])
                print(f"Session {session.session_id}, Turn {turn_idx}: {content_length} characters")
```

### Correlating Inputs with Results

Combine `artifacts/my-run/inputs.json` with `artifacts/my-run/profile_export.jsonl` for deeper analysis:

```python
from pathlib import Path
from aiperf.common.models import InputsFile, MetricRecordInfo
import json

def correlate_inputs_and_results(inputs_path: Path, results_path: Path):
    """Correlate input prompts with performance metrics."""
    # Load inputs
    with open(inputs_path, encoding="utf-8") as f:
        inputs = InputsFile.model_validate(json.load(f))

    # Create session lookup
    session_inputs = {session.session_id: session for session in inputs.data}

    # Process results and correlate
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = MetricRecordInfo.model_validate_json(line)

                # Skip failed requests
                if record.error is not None:
                    continue

                # Find corresponding input
                conv_id = record.metadata.conversation_id
                if conv_id in session_inputs:
                    session = session_inputs[conv_id]
                    turn_idx = record.metadata.turn_index

                    if turn_idx < len(session.payloads):
                        payload = session.payloads[turn_idx]
                        input_tokens = record.metrics.get("input_sequence_length", {}).get("value", 0)
                        latency = record.metrics.get("request_latency", {}).get("value", 0)

                        print(f"Conv {conv_id}, Turn {turn_idx}: "
                              f"{input_tokens} tokens, {latency:.2f}ms latency")

correlate_inputs_and_results(
    Path("artifacts/my-run/inputs.json"),
    Path("artifacts/my-run/profile_export.jsonl")
)
```

## Best Practices

### Data Validation

Always use Pydantic models for type-safe parsing and automatic validation:

```python
from aiperf.common.models import MetricRecordInfo, InputsFile

# Recommended: Type-safe parsing with validation
record = MetricRecordInfo.model_validate_json(line)
inputs = InputsFile.model_validate_json(f.read())
```
