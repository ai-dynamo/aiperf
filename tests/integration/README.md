<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Integration Tests

End-to-end tests for AIPerf against a [FakeAI](https://github.com/ajcasagrande/FakeAI) server.

## Quick Start

```bash
make test-integration          # Run all tests (parallel)
make test-integration-verbose  # Sequential with live output
```

## Architecture

### Core Components

**[conftest.py](conftest.py)** - Test configuration and fixtures
- `cli: AIPerfCLI` - Command runner that returns `AIPerfResults` with parsed outputs
- `fakeai_server: FakeAIServer` - Mock server instance with `url` and `dcgm_url`
- `tmp_path: Path` - Auto-managed temp directories for test artifacts

**[test_integration.py](test_integration.py)** - Comprehensive test suite organized by test classes, covering all AIPerf features including endpoints, streaming, multimodal inputs, performance testing, output formats, and metric validation. See the file's module docstring for the complete organization.

**[models.py](models.py)** - Data models
- `AIPerfResults` - Main results wrapper with Pydantic model properties
- `AIPerfSubprocessResult` - Process execution result
- `FakeAIServer` - Server connection info
- `VideoDetails` - Video metadata
- `AIPerfCLI` - CLI wrapper for running commands

**[metric_validators.py](metric_validators.py)** - Metric validation helpers
- `extract_metric_values()` - Extract metric from JSONL records
- `compute_stats()` - Compute aggregate statistics
- `validate_metric_stats()` - Validate single metric
- `validate_all_metrics()` - Validate all metrics at once

**[utils.py](utils.py)** - Test utilities
- `create_rankings_dataset()` - Generate rankings test data
- `extract_base64_video_details()` - Parse video metadata

## Writing Tests

All tests use async pytest with the integration marker:

```python
@pytest.mark.integration
@pytest.mark.asyncio
class TestMyFeature:
    async def test_example(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Test description."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        # Assertions
        assert result.request_count == 10
        assert result.has_streaming_metrics
```

### Test Markers

- `@pytest.mark.integration` - Required for all integration tests (deselected by default)
- `@pytest.mark.performance` - For high-load/stress tests (deselected by default)
- `@pytest.mark.ffmpeg` - For tests requiring FFmpeg binary (deselected by default)

**Default behavior:**
- Unit tests run by default
- Integration, performance, and ffmpeg tests are skipped by default
- To run marked tests, explicitly use `-m` flag

**Common usage:**

```bash
# Run default tests (unit tests only)
pytest

# Run integration tests
pytest -m integration

# Run integration tests but skip ffmpeg tests
pytest -m "integration and not ffmpeg"

# Run only ffmpeg tests
pytest -m ffmpeg

# Run everything (unit + integration + performance + ffmpeg)
pytest -m ""
```

**IDE Test Runners:**

When you run tests from your IDE (VSCode, PyCharm, etc.):
- **VSCode:** The project's [.vscode/settings.json](../../.vscode/settings.json) is configured to respect the same marker filtering as the CLI
- **Other IDEs:** You may need to configure pytest arguments manually to include `-m "not integration and not performance and not ffmpeg"`
- **Running individual tests:** When you click to run a specific test function, marker filtering still applies (test will be skipped if markers don't match)
- **To run marked tests in IDE:** Edit `.vscode/settings.json` and change the `-m` argument or remove it entirely

## Working with Results

The `AIPerfResults` object provides parsed outputs via Pydantic models:

```python
# Parsed outputs (Pydantic models)
result.json      # JsonExportData - aggregate metrics
result.inputs    # InputsFile - test payloads
result.jsonl     # List[MetricRecordInfo] - raw records
result.csv       # CSV export as string
result.log       # Log file content

# Convenience properties
result.request_count          # Completed requests
result.has_all_outputs        # All files exist
result.has_streaming_metrics  # TTFT, ITL present
result.has_input_images       # Images in inputs
result.has_gpu_telemetry      # GPU metrics collected

# Direct metric access
result.json.request_latency.avg
result.json.time_to_first_token.p99
```

## Validating Metrics

Use `validate_all_metrics()` to verify aggregate metrics match raw data:

```python
from tests.integration.metric_validators import validate_all_metrics

# Validate all metrics at once
computed = validate_all_metrics(result.jsonl, result.json)

# Returns dict[metric_name, ComputedStats] with all statistics
assert "request_latency" in computed
assert computed["request_latency"].count == result.request_count
```

## Tips

- Use `--ui simple` for faster tests (no interactive dashboard)
- Keep request counts low (10-100) for faster iteration
- Use `validate_all_metrics()` to ensure data integrity
- Check `result.log` when tests fail for detailed error messages
- All outputs are automatically validated as Pydantic models
