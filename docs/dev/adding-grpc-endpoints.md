<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adding New gRPC Endpoints

This guide explains how to add support for new gRPC-based inference protocols in AIPerf. The gRPC transport is fully generic and protocol-agnostic — all proto knowledge is isolated in pluggable **serializer** classes, so you never need to modify the transport or client.

## Architecture Overview

AIPerf separates **endpoints** (payload formatting and response parsing), **transports** (wire protocol and connection management), and **serializers** (proto-specific byte conversion):

```
InferenceClient
  |
  |-- Endpoint (format_payload / parse_response)
  |     Converts RequestInfo <-> protocol-specific dict
  |
  |-- GrpcTransport (send_request)
  |     Uses serializer to convert dict <-> bytes. Protocol-agnostic.
  |     |
  |     |-- Serializer (serialize_request / deserialize_response / deserialize_stream_response)
  |     |     Converts dict <-> protobuf bytes. Proto-specific.
  |     |
  |     |-- GenericGrpcClient (unary / server_stream)
  |           Sends/receives raw bytes over gRPC channel. Proto-free.
```

**Key contracts:**
- **Endpoints** produce and consume `dict[str, Any]` payloads. They never import protobuf types.
- **Serializers** convert between dicts and protobuf bytes. They are the ONLY layer that imports proto stubs.
- **GrpcTransport** and **GenericGrpcClient** operate on raw `bytes`. They never import proto stubs.

### What's Protocol-Specific vs Generic

| Component | Protocol-Specific | Generic (Reusable) |
|---|---|---|
| Proto definitions (`.proto`) | Yes | - |
| Serializer class (dict <-> protobuf bytes) | Yes | - |
| `payload_converter.py` (dict <-> protobuf objects) | Yes | - |
| Endpoint class (`format_payload` / `parse_response`) | Yes | - |
| `GrpcTransport` (timing, tracing, cancellation) | - | Yes |
| `GenericGrpcClient` (raw bytes over gRPC) | - | Yes |
| `grpc_defaults.py` (channel options) | - | Yes |
| `status_mapping.py` (gRPC -> HTTP codes) | - | Yes |
| `trace_data.py` (GrpcTraceData) | - | Yes |
| `stream_chunk.py` (StreamChunk) | - | Yes |
| Plugin registry, transport detection | - | Yes |

### How the Serializer is Loaded

The gRPC transport loads its serializer dynamically from endpoint metadata in `plugins.yaml`:

```yaml
endpoint:
  kserve_v2_infer:
    class: aiperf.endpoints.kserve_v2_infer:KServeV2InferEndpoint
    metadata:
      grpc:
        serializer: aiperf.transports.grpc.kserve_v2_serializers:KServeV2GrpcSerializer
        method: /inference.GRPCInferenceService/ModelInfer
        stream_method: /inference.GRPCInferenceService/ModelStreamInfer
```

At init, `GrpcTransport._init_serializer()` reads the endpoint's `grpc` metadata, imports the serializer class via `importlib`, and stores the method paths. This means:
- Adding a new gRPC protocol = new serializer + new endpoint + `plugins.yaml` entry
- No transport code changes required

## Extension Strategies

### Strategy A: New Endpoint, Same Protocol

**When:** The server speaks KServe V2 gRPC but you need different payload formatting (e.g., different tensor layouts, custom parameter conventions).

**What to create:** Just a new endpoint class. It reuses the existing `KServeV2GrpcSerializer`.

```python
# src/aiperf/endpoints/my_v2_endpoint.py
from __future__ import annotations

from typing import Any

from aiperf.common.models import InferenceServerResponse, ParsedResponse, RequestInfo
from aiperf.endpoints.base_endpoint import BaseEndpoint


class MyV2Endpoint(BaseEndpoint):
    """Custom V2 endpoint with different tensor layout."""

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        turn = request_info.turns[0]
        prompt = " ".join(
            content for text in turn.texts for content in text.contents if content
        )
        return {
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [prompt],
                },
                {
                    "name": "PRIORITY",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [1],
                },
            ],
            "parameters": {"max_tokens": 512, "temperature": 0.7},
        }

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        json_obj = response.get_json()
        if not json_obj:
            return None
        outputs = json_obj.get("outputs", [])
        for output in outputs:
            data = output.get("data")
            if data and isinstance(data, list) and data[0]:
                return ParsedResponse(
                    perf_ns=response.perf_ns,
                    data=self.make_text_response_data(str(data[0])),
                )
        return None
```

Register in `plugins.yaml`, pointing to the same KServe V2 serializer:

```yaml
endpoint:
  my_v2_endpoint:
    class: aiperf.endpoints.my_v2_endpoint:MyV2Endpoint
    description: Custom V2 endpoint with priority scheduling
    metadata:
      endpoint_path: /v2/models/{model_name}/infer
      supports_streaming: true
      produces_tokens: true
      metrics_title: My V2 Metrics
      service_kind: kserve
      grpc:
        serializer: aiperf.transports.grpc.kserve_v2_serializers:KServeV2GrpcSerializer
        method: /inference.GRPCInferenceService/ModelInfer
        stream_method: /inference.GRPCInferenceService/ModelStreamInfer
```

Usage: `aiperf profile --endpoint-type my_v2_endpoint --url grpc://triton:8001 ...`

### Strategy B: New gRPC Protocol (New Serializer)

**When:** The server uses a different `.proto` definition (different service name, different message types, different RPC methods).

**What to create:** Proto stubs, a payload converter, a serializer class, and an endpoint. You do NOT need a new transport or client — `GrpcTransport` and `GenericGrpcClient` are reused as-is.

#### Step 1: Define the Proto and Generate Stubs

Create your `.proto` file and generate stubs:

```
src/aiperf/transports/grpc/proto/my_service.proto
src/aiperf/transports/grpc/proto/my_service_pb2.py      (generated)
src/aiperf/transports/grpc/proto/my_service_pb2_grpc.py  (generated)
```

Follow the pattern in `tools/generate_grpc_stubs.py` for stub generation.

Add generated files to the ruff exclude in `pyproject.toml`:

```toml
exclude = ["...", "src/aiperf/transports/grpc/proto/my_service_pb2*.py"]
```

#### Step 2: Create a Payload Converter

The converter translates between the endpoint's dict payload and your protobuf messages:

```python
# src/aiperf/transports/grpc/my_payload_converter.py
from __future__ import annotations

from typing import Any

from aiperf.transports.grpc.proto import my_service_pb2 as pb2


def dict_to_my_request(
    payload: dict[str, Any],
    model_name: str,
    request_id: str = "",
) -> pb2.MyInferRequest:
    """Convert endpoint dict -> MyInferRequest protobuf."""
    request = pb2.MyInferRequest()
    request.model_name = model_name
    request.prompt = payload["prompt"]
    request.max_tokens = payload.get("max_tokens", 256)
    return request


def my_response_to_dict(response: pb2.MyInferResponse) -> dict[str, Any]:
    """Convert MyInferResponse protobuf -> dict for endpoint parsing."""
    return {
        "text": response.generated_text,
        "finish_reason": response.finish_reason,
    }
```

#### Step 3: Create the Serializer

The serializer wraps your payload converter and implements `GrpcSerializerProtocol`. This is the ONLY file (besides the payload converter) that imports proto stubs.

```python
# src/aiperf/transports/grpc/my_serializer.py
from __future__ import annotations

from typing import Any

from aiperf.transports.grpc.my_payload_converter import (
    dict_to_my_request,
    my_response_to_dict,
)
from aiperf.transports.grpc.proto import my_service_pb2 as pb2
from aiperf.transports.grpc.stream_chunk import StreamChunk


class MyGrpcSerializer:
    """gRPC serializer for MyService protocol.

    Implements GrpcSerializerProtocol. Discovered via plugins.yaml
    endpoint metadata (grpc.serializer).
    """

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        """Convert a dict payload to serialized MyInferRequest bytes."""
        proto = dict_to_my_request(
            payload, model_name=model_name, request_id=request_id
        )
        return proto.SerializeToString()

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        """Deserialize MyInferResponse bytes to a dict and wire size."""
        response = pb2.MyInferResponse()
        response.ParseFromString(data)
        return my_response_to_dict(response), len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        """Deserialize streaming response bytes to a StreamChunk."""
        stream_resp = pb2.MyStreamResponse()
        stream_resp.ParseFromString(data)

        if stream_resp.error_message:
            return StreamChunk(
                error_message=stream_resp.error_message,
                response_dict=None,
                response_size=len(data),
            )

        resp_dict = my_response_to_dict(stream_resp.response)
        return StreamChunk(
            error_message=None,
            response_dict=resp_dict,
            response_size=len(data),
        )
```

**Protocol conformance:** Your serializer must implement these three methods with matching signatures. The `GrpcSerializerProtocol` in `grpc_transport.py` is `@runtime_checkable`, so you can verify conformance in tests:

```python
from aiperf.transports.grpc.grpc_transport import GrpcSerializerProtocol

def test_implements_protocol():
    assert isinstance(MyGrpcSerializer(), GrpcSerializerProtocol)
```

#### Step 4: Create the Endpoint

```python
# src/aiperf/endpoints/my_endpoint.py
from __future__ import annotations

from typing import Any

from aiperf.common.models import InferenceServerResponse, ParsedResponse, RequestInfo
from aiperf.endpoints.base_endpoint import BaseEndpoint


class MyEndpoint(BaseEndpoint):
    """Endpoint for MyService protocol."""

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        turn = request_info.turns[0]
        prompt = " ".join(
            content for text in turn.texts for content in text.contents if content
        )
        return {
            "prompt": prompt,
            "max_tokens": turn.max_tokens or 256,
        }

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        json_obj = response.get_json()
        if not json_obj:
            return None
        text = json_obj.get("text")
        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=self.make_text_response_data(text),
        ) if text else None
```

#### Step 5: Register in plugins.yaml

```yaml
endpoint:
  my_endpoint:
    class: aiperf.endpoints.my_endpoint:MyEndpoint
    description: MyService inference endpoint
    metadata:
      endpoint_path: null
      supports_streaming: true
      produces_tokens: true
      metrics_title: MyService Metrics
      grpc:
        serializer: aiperf.transports.grpc.my_serializer:MyGrpcSerializer
        method: /mypackage.MyService/Infer
        stream_method: /mypackage.MyService/StreamInfer
```

The `method` and `stream_method` are fully-qualified gRPC method paths matching your `.proto` service definition. Set `stream_method` to `null` if the protocol doesn't support streaming.

Then regenerate plugin artifacts:

```bash
uv run ./tools/generate_plugin_artifacts.py
make validate-plugin-schemas
```

#### Step 6: Usage

```bash
aiperf profile \
    --model my-model \
    --url grpc://server:9000 \
    --endpoint-type my_endpoint \
    --request-count 50
```

No new URL schemes or transport registration needed — the existing `grpc://` / `grpcs://` schemes route to `GrpcTransport`, which loads your serializer from the endpoint metadata.

### Strategy C: New Transport (Different Wire Protocol)

**When:** You need a completely different wire protocol that isn't standard gRPC (e.g., custom framing, non-protobuf serialization, or a transport that doesn't use `grpc.aio.Channel`).

This is rare. For standard gRPC with a different `.proto`, use Strategy B. You only need a new transport if the underlying connection management differs.

See the [Transport Pattern](./patterns.md#transport-pattern) for the `BaseTransport` interface and registration.

## The GrpcSerializerProtocol

The serializer protocol is defined in `grpc_transport.py`:

```python
@runtime_checkable
class GrpcSerializerProtocol(Protocol):
    def serialize_request(
        self, payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes: ...

    def deserialize_response(self, data: bytes) -> tuple[dict[str, Any], int]: ...

    def deserialize_stream_response(self, data: bytes) -> StreamChunk: ...
```

| Method | Input | Output | Purpose |
|---|---|---|---|
| `serialize_request` | dict payload, model name, request ID | `bytes` | Convert endpoint dict to protobuf wire bytes |
| `deserialize_response` | `bytes` | `(dict, int)` | Convert unary response bytes to dict + wire size |
| `deserialize_stream_response` | `bytes` | `StreamChunk` | Convert streaming chunk bytes to `StreamChunk` |

### StreamChunk

`StreamChunk` is a protocol-agnostic container for streaming responses:

```python
@dataclasses.dataclass(frozen=True, slots=True)
class StreamChunk:
    error_message: str | None          # Set if this chunk is an error
    response_dict: dict[str, Any] | None  # Set if this chunk has data
    response_size: int                 # Wire size in bytes
```

If `error_message` is set, the transport treats it as an error and stops streaming. Otherwise, `response_dict` is serialized to JSON and stored as a `TextResponse`.

### plugins.yaml grpc Metadata Block

```yaml
grpc:
  serializer: module.path:ClassName    # Required. Class implementing GrpcSerializerProtocol.
  method: /service/Method              # Required. Fully-qualified unary RPC path.
  stream_method: /service/StreamMethod # Optional. Fully-qualified streaming RPC path.
```

If `stream_method` is `null`, the endpoint does not support `--streaming` over gRPC.

## Reusable Components

When building a new gRPC serializer, these components can be reused directly:

| Component | File | What It Provides |
|---|---|---|
| `GrpcTransport` | `grpc_transport.py` | Generic transport (timing, tracing, cancellation, metadata) |
| `GenericGrpcClient` | `grpc_client.py` | Proto-free gRPC client (raw bytes) |
| `StreamChunk` | `stream_chunk.py` | Protocol-agnostic streaming response container |
| `GrpcSerializerProtocol` | `grpc_transport.py` | Runtime-checkable serializer interface |
| `DEFAULT_CHANNEL_OPTIONS` | `grpc_defaults.py` | Tuned channel options (message size, keepalive) |
| `grpc_status_to_http()` | `status_mapping.py` | gRPC -> HTTP status code mapping |
| `GrpcTraceData` | `trace_data.py` | Trace data with gRPC status fields |
| `BaseTransport` | `base_transports.py` | Lifecycle, header building, URL handling |
| `BaseEndpoint` | `endpoints/base_endpoint.py` | Response auto-detection, text/embedding/ranking extraction |
| `AIPerfLoggerMixin` | `common/mixins/` | Structured logging |

## Checklist

When adding a new gRPC endpoint with a new protocol:

- [ ] Proto file created and stubs generated
- [ ] Generated stubs excluded from ruff in `pyproject.toml`
- [ ] Payload converter: `dict -> protobuf` and `protobuf -> dict`
- [ ] Serializer class implementing `GrpcSerializerProtocol`
- [ ] Endpoint extending `BaseEndpoint` with `format_payload` / `parse_response`
- [ ] Endpoint registered in `plugins.yaml` with `grpc` metadata block
- [ ] `uv run ./tools/generate_plugin_artifacts.py` regenerates enums
- [ ] `make validate-plugin-schemas` passes
- [ ] Unit tests for serializer (roundtrip + protocol conformance) and endpoint
- [ ] `uv run pytest tests/unit/ -n auto` passes

When adding a new endpoint reusing an existing protocol (Strategy A):

- [ ] Endpoint extending `BaseEndpoint` with `format_payload` / `parse_response`
- [ ] Endpoint registered in `plugins.yaml` with existing `grpc` metadata block
- [ ] `uv run ./tools/generate_plugin_artifacts.py` regenerates enums
- [ ] `make validate-plugin-schemas` passes
- [ ] Unit tests for endpoint
- [ ] `uv run pytest tests/unit/ -n auto` passes

## Related

- [gRPC Transport Guide](../tutorials/grpc-transport.md) -- User-facing tutorial
- [Code Patterns](./patterns.md) -- Transport and Trace Data patterns
- [Plugin System](../plugins/plugin-system.md) -- Plugin registration details
- [Source: GrpcTransport](../../src/aiperf/transports/grpc/grpc_transport.py) -- Generic transport implementation
- [Source: GenericGrpcClient](../../src/aiperf/transports/grpc/grpc_client.py) -- Proto-free gRPC client
- [Source: KServeV2GrpcSerializer](../../src/aiperf/transports/grpc/kserve_v2_serializers.py) -- Reference serializer implementation
- [Source: payload_converter](../../src/aiperf/transports/grpc/payload_converter.py) -- V2 dict/protobuf conversion
- [Source: InferenceClient](../../src/aiperf/workers/inference_client.py) -- Transport/endpoint wiring
