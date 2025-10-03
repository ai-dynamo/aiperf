<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Examples

Examples showing how to load and work with AIPerf data files.

## Profile Export Format

AIPerf generates `profile_export.jsonl` files in JSON Lines format. Each line is a complete JSON object representing one metric record.

### File Structure

```json
{
  "record_id": "",
  "metadata": {
    "conversation_id": "uuid-string",
    "turn_index": 0,
    "timestamp_ns": 1234567890123456789,
    "worker_id": "worker_abc123",
    "record_processor_id": "processor_xyz789",
    "credit_phase": "profiling",
    "error": null
  },
  "metrics": {
    "request_latency": {"value": 123.45, "unit": "ms"},
    "ttft": {"value": 45.67, "unit": "ms"},
    "output_sequence_length": {"value": 100, "unit": "tokens"}
  }
}
```

### Data Models

AIPerf uses Pydantic models for type-safe data handling:

```python
from aiperf.common.models.record_models import (
    MetricRecordInfo,      # Top-level record
    MetricRecordMetadata,  # Request metadata
    MetricValue,           # Individual metric with value and unit
)
```

## Loading Data

### Sync Version

Simple synchronous loading:

```python
from pathlib import Path
from aiperf.common.models.record_models import MetricRecordInfo

def load_records(file_path: Path) -> list[MetricRecordInfo]:
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = MetricRecordInfo.model_validate_json(line)
                records.append(record)
    return records

records = load_records(Path("artifacts/run/profile_export.jsonl"))
```

**Run the example:**
```bash
python examples/parse_profile_export.py artifacts/run/profile_export.jsonl
```

### Async Version

Async loading with `aiofiles`:

```python
import aiofiles
import asyncio
from pathlib import Path
from aiperf.common.models.record_models import MetricRecordInfo

async def load_records_async(file_path: Path) -> list[MetricRecordInfo] :
    records = []
    async with aiofiles.open(file_path, encoding="utf-8") as f:
        async for line in f:
            if line.strip():
                record = MetricRecordInfo.model_validate_json(line)
                records.append(record)
    return records

# Usage
records = await load_records_async(Path("profile_export.jsonl"))
```

**Run the example:**
```bash
python examples/parse_profile_export.py artifacts/run/profile_export.jsonl --async
```

**Load multiple files concurrently:**
```python
async def load_multiple_files(file_paths: list[Path]):
    tasks = [load_records_async(path) for path in file_paths]
    results = await asyncio.gather(*tasks)
    return dict(zip(file_paths, results))

files = [Path("run1/profile_export.jsonl"), Path("run2/profile_export.jsonl")]
results = await load_multiple_files(files)
```

## Accessing Data

Once loaded, records are fully typed Pydantic models:

```python
# Iterate over records
for record in records:
    print(record.record_id)
    print(record.metadata.worker_id)
    print(record.metadata.timestamp_ns)

# Access metrics
for record in records:
    if 'request_latency' in record.metrics:
        latency = record.metrics['request_latency']
        print(f"Latency: {latency.value} {latency.unit}")

# Filter by success/failure
successful = [r for r in records if r.metadata.error is None]
failed = [r for r in records if r.metadata.error is not None]

# Get all available metrics
all_metrics = set()
for record in records:
    all_metrics.update(record.metrics.keys())
print(f"Available metrics: {all_metrics}")
```

## Converting to Other Formats

### To Dictionary

```python
# Full export
data = [record.model_dump() for record in records]

# Custom format
data = [
    {
        'record_id': record.record_id,
        'worker_id': record.metadata.worker_id,
        'timestamp': record.metadata.timestamp_ns,
        **{name: metric.value for name, metric in record.metrics.items()}
    }
    for record in records
]
```

### To Pandas DataFrame

```python
import pandas as pd

def to_dataframe(records: list[MetricRecordInfo]) -> pd.DataFrame:
    data = []
    for record in records:
        row = {
            'record_id': record.record_id,
            'worker_id': record.metadata.worker_id,
            'timestamp_ns': record.metadata.timestamp_ns,
            'has_error': record.metadata.error is not None,
        }
        # Add scalar metrics as columns
        for name, metric in record.metrics.items():
            if not isinstance(metric.value, list):
                row[name] = metric.value
        data.append(row)

    return pd.DataFrame(data)

df = to_dataframe(records)
```

### To Polars DataFrame

```python
import polars as pl

def to_polars(records: list[MetricRecordInfo]) -> pl.DataFrame:
    data = [
        {
            'record_id': record.record_id,
            'worker_id': record.metadata.worker_id,
            'timestamp_ns': record.metadata.timestamp_ns,
            **{
                name: metric.value
                for name, metric in record.metrics.items()
                if not isinstance(metric.value, list)
            }
        }
        for record in records
    ]
    return pl.DataFrame(data)

df = to_polars(records)
```

## Common Metrics

Typical metrics you'll find in the data:

| Metric | Description | Unit | Type |
|--------|-------------|------|------|
| `request_latency` | Total request time | ms | float |
| `ttft` | Time to first token | ms | float |
| `tpot` | Time per output token | ms | float |
| `inter_token_latency` | Inter-token latency | ms | float |
| `e2e_latency` | End-to-end latency | ms | float |
| `input_sequence_length` | Input tokens | tokens | int |
| `output_sequence_length` | Output tokens | tokens | int |
| `request_count` | Request count | requests | int |
| `error_request_count` | Failed requests | requests | int |

## Type Information

The Pydantic models provide full type information:

```python
from aiperf.common.models.record_models import MetricRecordInfo

# Get type hints
print(MetricRecordInfo.model_fields.keys())
# dict_keys(['record_id', 'metadata', 'metrics'])

# Access model schema
schema = MetricRecordInfo.model_json_schema()

# Validate data
try:
    record = MetricRecordInfo.model_validate(data)
except ValidationError as e:
    print(f"Invalid data: {e}")
```

## Requirements

- Python 3.10+
- `aiperf` (already installed)
- `rich` (for pretty output)
- `aiofiles` (for async version only)

Install additional dependencies:
```bash
pip install rich aiofiles
```
