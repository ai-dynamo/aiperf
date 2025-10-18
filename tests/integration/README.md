<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Integration Tests

End-to-end tests for AIPerf that validate real-world scenarios against a mock LLM server.

## Test Style

Tests follow a simple pattern:

1. Pass your AIPerf CLI command as a string to `cli.run()`
2. Await the result to get an `AIPerfResults` object
3. Make assertions against the result properties

```python
@pytest.mark.integration
@pytest.mark.asyncio
class TestChatEndpoint:
    async def test_basic_chat(self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer):
        """Basic chat completion test."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen3-0.6B \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10
```

## Setup

Install AIPerf and the mock server:

```bash
make first-time-setup
```

## Running Tests

```bash
# Run all integration tests (parallel)
make integration-tests

# Run all integration tests (verbose, sequential with live AIPerf output)
make integration-tests-verbose
```


## Key Components

**Fixtures (conftest.py)**
- `aiperf_mock_server: AIPerfMockServer` - Mock LLM server instance
- `cli: AIPerfCLI` - CLI wrapper for running AIPerf commands

**Models (models.py)**
- `AIPerfResults` - Result wrapper with typed properties for all output artifacts
- `AIPerfMockServer` - Server connection info
- `AIPerfSubprocessResult` - Subprocess execution result
