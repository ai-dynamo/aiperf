<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Integration Tests

End-to-end integration tests for AIPerf that validate real-world scenarios against a [FakeAI](https://github.com/ajcasagrande/FakeAI) server. Tests are organized by API endpoints and features, with a focus on clarity, maintainability, and fast execution.

## Test Style

All tests are organized into test classes and follow a consistent pattern:

1. Pass your AIPerf CLI command as a string to `cli.run()`
2. Await the result to get an `AIPerfResults` object
3. Make assertions against the result properties

```python
@pytest.mark.integration
@pytest.mark.asyncio
class TestChatEndpoint:
    async def test_basic_chat_endpoint(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Basic chat endpoint test."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen3-0.6B \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --concurrency 10 \
                --request-count 100 \
                --streaming \
                --ui simple
            """
        )
        assert result.request_count == 100
        assert result.has_streaming_metrics
```

## Running Tests

```bash
# Run all integration tests (parallel)
make test-integration

# Run all integration tests (verbose, sequential with AIPerf output)
make test-integration-verbose

# Run all endpoint tests
make test-integration tests/integration/endpoints/

# Run all feature tests
make test-integration tests/integration/features/

# Run specific endpoint tests
make test-integration tests/integration/endpoints/test_chat.py

# Run specific test class
make test-integration tests/integration/endpoints/test_chat.py::TestChatEndpoint

# Run specific test
make test-integration tests/integration/endpoints/test_chat.py::TestChatEndpoint::test_basic_chat
```

## Test Organization

Tests are organized by `endpoints` and `features`:

- **Endpoint tests** - Tests for specific API endpoints (chat, completions, embeddings, rankings). Each test class focuses on the behavior and capabilities of a single endpoint.
- **Feature tests** - Tests for AIPerf features that are endpoint agnostic or focus on specific behaviors (dashboard, deterministic mode, output formats, synthetic data).

## Key Components

### Fixtures (conftest.py)
- `fakeai_server` - Mock AI server
- `cli` - CLI wrapper for running benchmarks

### Helpers (helpers.py)
- `AIPerfSubprocessResult` - Subprocess result
- `FakeAIServer` - Server connection info
- `AIPerfResults` - Simple result wrapper with properties
- `AIPerfCLI` - CLI wrapper that runs commands and returns results
