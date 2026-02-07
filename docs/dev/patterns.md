<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Code Patterns

Code examples for common development tasks. Referenced from CLAUDE.md.

## Service Pattern

Services run in separate processes via `bootstrap.py`:

```python
class MyService(BaseComponentService):
    @on_message(MessageType.MY_MSG)
    async def _handle(self, msg: MyMsg) -> None:
        await self.publish(ResponseMsg(data=msg.data))
```

Register in `plugins.yaml`:

```yaml
service:
  my_service:
    class: aiperf.my_module.my_service:MyService
    description: My custom service
    metadata:
      required: true
      auto_start: true
```

**Config types:**
- `ServiceConfig`: infrastructure (ZMQ ports, logging level)
- `UserConfig`: benchmark params (endpoints, loadgen settings)

## Model Pattern

Use `AIPerfBaseModel` for data, `BaseConfig` for configuration:

```python
from pydantic import Field
from aiperf.common.models import AIPerfBaseModel

class Record(AIPerfBaseModel):
    ts_ns: int = Field(description="Timestamp in nanoseconds")
    value: float = Field(description="Measured value")
```

## Message Pattern

Messages require `message_type` field and handler decorator:

```python
from aiperf.common.messages import Message
from aiperf.common.hooks import on_message

class MyMsg(Message):
    message_type: MessageType = MessageType.MY_MSG
    data: list[Record] = Field(description="Records to process")

# In service class:
@on_message(MessageType.MY_MSG)
async def _handle(self, msg: MyMsg) -> None:
    await self.publish(OtherMsg(data=msg.data))
```

Auto-subscription happens during `@on_init` phase.

## Plugin System Pattern

YAML-based registry with lazy-loading:

```yaml
# plugins.yaml
endpoint:
  chat:
    class: aiperf.endpoints.openai_chat:ChatEndpoint
    description: OpenAI Chat Completions endpoint
    metadata:
      endpoint_path: /v1/chat/completions
      supports_streaming: true
      produces_tokens: true
      tokenizes_input: true
      supports_audio: true
      supports_images: true
      supports_videos: true
      metrics_title: LLM Metrics
```

```python
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

EndpointClass = plugins.get_class(PluginType.ENDPOINT, 'chat')
```

## Error Handling Pattern

Log errors and publish `ErrorDetails` in messages:

```python
try:
    await risky_operation()
except Exception as e:
    self.error(f"Operation failed: {e!r}")
    await self.publish(ResultMsg(error=ErrorDetails.from_exception(e)))
```

## Logging Pattern

Use lambda for expensive log messages:

```python
# Expensive - lambda defers evaluation
self.debug(lambda: f"Processing {len(self._items())} items")

# Cheap - direct string is fine
self.info("Starting service")
```

## Testing Pattern

```python
import pytest
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from tests.harness import mock_plugin

@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_func()
    assert result.status == "ok"

@pytest.mark.parametrize("input,expected",
    [
        ("a", 1),
        ("b", 2),
    ]
)  # fmt: skip
def test_with_params(input, expected):
    assert process(input) == expected

def test_with_mock_plugin():
    with mock_plugin(PluginType.ENDPOINT, "test", MockClass):
        assert plugins.get_class(PluginType.ENDPOINT, "test") == MockClass
```

**Auto-fixtures** (always active): asyncio.sleep runs instantly, RNG=42, singletons reset.

## Transport Pattern

Transports extend `BaseTransport` (which uses `AIPerfLifecycleMixin`) and manage request delivery:

```python
from typing import Any

from aiperf.common.hooks import on_init, on_stop
from aiperf.common.models import RequestInfo, RequestRecord
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import BaseTransport, FirstTokenCallback

class MyTransport(BaseTransport):
    @classmethod
    def metadata(cls) -> TransportMetadata:
        return TransportMetadata(
            transport_type="my_protocol",
            url_schemes=["myproto", "myprotos"],
        )

    @on_init
    async def _init_client(self) -> None:
        # Parse URL, create client connection
        base_url = self.model_endpoint.endpoint.base_url
        ...

    @on_stop
    async def _close_client(self) -> None:
        # Close client connection
        ...

    def get_url(self, request_info: RequestInfo) -> str:
        # Return target URL for request
        ...

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        return {}  # Transport-specific headers

    async def send_request(
        self, request_info: RequestInfo, payload: dict[str, Any],
        *, first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        # Send request, populate record with responses and timing
        ...
```

Register in `plugins.yaml`:

```yaml
transport:
  my_protocol:
    class: aiperf.transports.my_transport:MyTransport
    description: My custom transport
    metadata:
      transport_type: my_protocol
      url_schemes: [myproto, myprotos]
```

Transport is auto-detected from URL scheme by `detect_transport_from_url()` in `workers/inference_client.py`.

## Trace Data Pattern

Custom trace data extends `BaseTraceData` and `TraceDataExport`. **Convention**: define the Export class before the Data class in the same file for readability and consistency with existing patterns (e.g., `GrpcTraceDataExport` before `GrpcTraceData`).

```python
from typing import Literal
from pydantic import Field
from aiperf.common.models.trace_models import BaseTraceData, TraceDataExport

# Convention: Export class defined before Data class
class MyTraceDataExport(TraceDataExport):
    trace_type: Literal["my_protocol"] = "my_protocol"
    my_field: int | None = Field(default=None, description="Protocol-specific field.")

class MyTraceData(BaseTraceData):
    trace_type: str = "my_protocol"
    my_field: int | None = Field(default=None, description="Protocol-specific field.")
```

The `trace_type` discriminator enables `to_export()` to automatically produce the correct export subclass. Base fields (`request_send_start_perf_ns`, `response_chunks`, etc.) are inherited.
